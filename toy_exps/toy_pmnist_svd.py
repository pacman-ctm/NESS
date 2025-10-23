import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from collections import OrderedDict
import time
import numpy as np
import argparse

# ---------- Data ----------
def make_permuted_mnist(root="./data", train_bs=64, test_bs=512, seed=0, n_tasks=5):
    """
    Create Permuted MNIST tasks where each task uses a different random permutation
    of the input pixels. Matches the implementation from the reference code.
    
    Task 1 uses the original order, Tasks 2-n use random permutations.
    Each task uses the same permutation for all samples.
    """
    from sklearn.utils import shuffle as sk_shuffle
    
    g = torch.Generator().manual_seed(seed)
    
    mean = (0.1307,)
    std = (0.3081,)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    tr_ds = datasets.MNIST(root, train=True, transform=tfm, download=True)
    te_ds = datasets.MNIST(root, train=False, transform=tfm, download=True)
    
    # Create permutation seeds for each task
    seeds = np.array(list(range(n_tasks)), dtype=int)
    
    permutations = []
    for i, r in enumerate(seeds):
        if i == 0:
            # Task 1: identity permutation (no change)
            permutations.append(np.arange(784))
        else:
            # Tasks 2-n: random permutations
            # Create a dummy array and shuffle it to get the permutation
            aux = np.arange(784)
            aux = sk_shuffle(aux, random_state=r * 100 + i)
            permutations.append(aux)
    
    def create_permuted_dataset(ds, permutation):
        """Apply permutation to the dataset"""
        # Get all data at once
        loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)
        data_list = []
        target_list = []
        
        for images, targets in loader:
            # Process each image
            for idx in range(images.size(0)):
                image = images[idx]
                # Flatten and apply permutation
                aux = image.view(-1)  # Shape: [784]
                aux = aux[permutation]  # Apply permutation by indexing
                image_perm = aux.view(1, 28, 28)  # Reshape back
                data_list.append(image_perm)
                target_list.append(targets[idx])
        
        x = torch.stack(data_list).squeeze(1)  # Shape: [N, 1, 28, 28]
        y = torch.stack(target_list)
        
        # Flatten for MLP input
        x = x.view(-1, 784)
        
        return TensorDataset(x, y)
    
    tasks = []
    for i, perm in enumerate(permutations):
        print(f"Creating Task {i+1} with permutation seed {seeds[i]}...")
        tr = create_permuted_dataset(tr_ds, perm)
        te = create_permuted_dataset(te_ds, perm)
        tasks.append({
            "name": f"Task {i+1} (pmnist-{seeds[i]})",
            "train": DataLoader(tr, batch_size=train_bs, shuffle=True, generator=g,
                                pin_memory=torch.cuda.is_available(), num_workers=2),
            "test":  DataLoader(te, batch_size=test_bs, shuffle=False,
                                pin_memory=torch.cuda.is_available(), num_workers=2),
            "permutation": perm,
            "seed": seeds[i]
        })
    
    return tasks

# ---------- Model ----------
class MLPNet(nn.Module):
    def __init__(self, n_hidden=100, n_outputs=10):
        super(MLPNet, self).__init__()
        self.act = OrderedDict()
        self.lin1 = nn.Linear(784, n_hidden, bias=False)
        self.lin2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.fc1  = nn.Linear(n_hidden, n_outputs, bias=False)
    
    def forward(self, x):
        self.act['Lin1'] = x
        x = self.lin1(x)        
        x = F.relu(x)
        self.act['Lin2'] = x
        x = self.lin2(x)        
        x = F.relu(x)
        self.act['fc1'] = x
        x = self.fc1(x)
        return x

# ---------- Training ----------
def train(model, tasks, device, epochs=10, lr=0.01, null_space_dim=100, hidden_lr_scale=0.01):
    criterion = nn.CrossEntropyLoss()
    
    # Store base weights after each task
    base_weights = {}
    
    # Initialize base weights from the model's initial weights
    for name, param in model.named_parameters():
        base_weights[name] = param.data.clone().to(device)
    
    # Store all previous task data
    all_previous_data = []
    
    results = []
    
    for task_id, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"Training on {task['name']}")
        print(f"{'='*60}")
        
        A = None  # Projection matrix
        
        # For task 2 onwards, perform SVD on all previous task data (X_1 to X_{i-1})
        if task_id > 0:
            print(f"\nPerforming SVD on data from all previous tasks (Task 1 to Task {task_id})...")
            
            # Combine all previous data: X_all = [X_1, X_2, ..., X_{i-1}]
            X_all = torch.cat(all_previous_data, dim=0)  # Shape: [N_total, 784]
            
            print(f"X_all shape (previous tasks only): {X_all.shape}")
            
            # Perform SVD
            X_all_cpu = X_all.cpu()
            U, S, Vt = torch.linalg.svd(X_all_cpu, full_matrices=False)
            
            # Move back to device
            S = S.to(device)
            Vt = Vt.to(device)
            
            print(f"SVD completed - U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")
            print(f"Singular values range: [{S[0].item():.4f}, {S[-1].item():.6f}]")
            
            # Get A: span of the LAST null_space_dim rows (null space)
            A = Vt[-null_space_dim:, :].clone()  # Shape: [null_space_dim, 784]
            print(f"A shape (null space projection): {A.shape}")
        
        # Collect current task data and add to all_previous_data for next tasks
        X_current = []
        for data, target in task['train']:
            X_current.append(data)
        X_current = torch.cat(X_current, dim=0).to(device)  # Shape: [N_i, 784]
        all_previous_data.append(X_current)
        
        # Initialize trainable parameters
        if task_id == 0:
            # Task 1: Train W_tilde directly (no projection)
            w_tilde = {}
            for name in base_weights.keys():
                w_tilde[name] = torch.zeros_like(base_weights[name], requires_grad=True, device=device)
            
            # Single optimizer with same learning rate for all
            optimizer = torch.optim.SGD(w_tilde.values(), lr=lr)
            trainable_params = w_tilde
        else:
            T = {}
            for name in base_weights.keys():
                weight_shape = base_weights[name].shape
                if name == 'lin1.weight':
                    T[name] = torch.zeros(weight_shape[0], null_space_dim, requires_grad=True, device=device)
                else:
                    # Hidden layers: regular parameterization
                    T[name] = torch.zeros_like(base_weights[name], requires_grad=True, device=device)
            
            optimizer = torch.optim.SGD([
                {'params': [T['lin1.weight']], 'lr': lr},
                {'params': [T['lin2.weight'], T['fc1.weight']], 'lr': lr * hidden_lr_scale}
            ])
            trainable_params = T
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(task['train']):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                # Manual forward pass
                x = data
                
                # Layer 1: lin1
                if task_id == 0:
                    w1 = base_weights['lin1.weight'] + trainable_params['lin1.weight']
                else:
                    # W_tilde = T @ A (projected to null space)
                    w_tilde_lin1 = torch.mm(trainable_params['lin1.weight'], A)
                    w1 = base_weights['lin1.weight'] + w_tilde_lin1
                x = F.linear(x, w1)
                x = F.relu(x)
                
                # Layer 2: lin2
                w2 = base_weights['lin2.weight'] + trainable_params['lin2.weight']
                x = F.linear(x, w2)
                x = F.relu(x)
                
                # Layer 3: fc1
                w3 = base_weights['fc1.weight'] + trainable_params['fc1.weight']
                output = F.linear(x, w3)
                
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                with torch.no_grad():
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            
            avg_loss = total_loss / len(task['train'])
            accuracy = 100. * correct / total
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
        
        # Update base weights: W_base = W_base + W_tilde
        with torch.no_grad():
            for name in base_weights.keys():
                if task_id == 0:
                    base_weights[name] = base_weights[name] + trainable_params[name].data
                else:
                    if name == 'lin1.weight':
                        w_tilde = torch.mm(trainable_params[name].data, A)
                        base_weights[name] = base_weights[name] + w_tilde
                    else:
                        base_weights[name] = base_weights[name] + trainable_params[name].data
            
            # Update model weights
            for name, param in model.named_parameters():
                param.data = base_weights[name].clone()
        
        # Evaluate on all tasks seen so far
        print(f"\nEvaluating after {task['name']}:")
        task_results = []
        for eval_task_id in range(task_id + 1):
            acc = evaluate(model, tasks[eval_task_id]['test'], device)
            task_results.append(acc)
            print(f"  {tasks[eval_task_id]['name']}: {acc:.2f}%")
        
        results.append(task_results)
    
    return results

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Permuted MNIST Continual Learning with Null Space Projection')
    
    # Data parameters
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Root directory for MNIST data (default: ./data)')
    parser.add_argument('--n-tasks', type=int, default=5,
                        help='Number of permuted MNIST tasks (default: 5)')
    parser.add_argument('--train-bs', type=int, default=64,
                        help='Training batch size (default: 64)')
    parser.add_argument('--test-bs', type=int, default=512,
                        help='Test batch size (default: 512)')
    
    # Model parameters
    parser.add_argument('--n-hidden', type=int, default=100,
                        help='Number of hidden units in each layer (default: 100)')
    parser.add_argument('--n-outputs', type=int, default=10,
                        help='Number of output classes (default: 10)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs per task (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--hidden-lr-scale', type=float, default=0.1,
                        help='Learning rate scale for hidden layers in tasks 2+ (default: 0.1)')
    
    # Continual learning parameters
    parser.add_argument('--null-space-dim', type=int, default=200,
                        help='Dimension of null space projection (default: 200)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Print configuration
    print(f"\n{'='*60}")
    print("Configuration:")
    print(f"{'='*60}")
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    print(f"{'='*60}\n")
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("Loading Permuted MNIST dataset...")
    tasks = make_permuted_mnist(root=args.data_root, train_bs=args.train_bs, 
                                test_bs=args.test_bs, seed=args.seed, n_tasks=args.n_tasks)
    print(f"Number of tasks: {len(tasks)}")
    print(f"Each task uses a different random permutation of the 784 input pixels\n")
    
    model = MLPNet(n_hidden=args.n_hidden, n_outputs=args.n_outputs).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters\n")
    
    # Train
    start_time = time.time()
    results = train(model, tasks, device, epochs=args.epochs, lr=args.lr, 
                   null_space_dim=args.null_space_dim, hidden_lr_scale=args.hidden_lr_scale)
    end_time = time.time()
    
    # Print final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Training completed in {end_time - start_time:.2f} seconds\n")
    
    print("Accuracy matrix (rows=after task, cols=task performance):")
    for i, task_accs in enumerate(results):
        print(f"After Task {i+1}: {' '.join([f'{acc:6.2f}%' for acc in task_accs])}")
    
    # Calculate average accuracy and forgetting
    if len(results) > 1:
        final_accs = results[-1]
        avg_acc = sum(final_accs) / len(final_accs)
        print(f"\nAverage Accuracy: {avg_acc:.2f}%")
        
        # Calculate forgetting
        forgetting = []
        for task_idx in range(len(tasks) - 1):
            max_acc = max([results[i][task_idx] for i in range(task_idx, len(results))])
            final_acc = results[-1][task_idx]
            forgetting.append(max_acc - final_acc)
        avg_forgetting = sum(forgetting) / len(forgetting) if forgetting else 0
        print(f"Average Forgetting: {avg_forgetting:.2f}%")

if __name__ == "__main__":
    main()