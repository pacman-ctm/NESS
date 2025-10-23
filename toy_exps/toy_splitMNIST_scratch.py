import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from collections import OrderedDict
import time

def make_split_mnist(root="./data", train_bs=64, test_bs=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    tfm = transforms.ToTensor()
    tr_ds = datasets.MNIST(root, train=True,  transform=tfm, download=True)
    te_ds = datasets.MNIST(root, train=False, transform=tfm, download=True)
    pairs = [(0,1),(2,3),(4,5),(6,7),(8,9)]
    def subset(ds, a, b):
        x, y = ds.data.float()/255.0, ds.targets
        m = (y==a)|(y==b)
        x = x[m].view(-1, 28*28)
        y = (y[m]==b).long()
        return TensorDataset(x, y)
    tasks = []
    for i,(a,b) in enumerate(pairs,1):
        tr = subset(tr_ds, a, b)
        te = subset(te_ds, a, b)
        tasks.append({
            "name": f"Task {i}: {a} vs {b}",
            "train": DataLoader(tr, batch_size=train_bs, shuffle=True, generator=g,
                                pin_memory=torch.cuda.is_available(), num_workers=2),
            "test":  DataLoader(te, batch_size=test_bs, shuffle=False,
                                pin_memory=torch.cuda.is_available(), num_workers=2),
        })
    return tasks

class MLPNet(nn.Module):
    def __init__(self, n_hidden=100, n_outputs=1):
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

def train(model, tasks, device, epochs=10, lr=0.01):
    """
    Continual learning training where W_current = W_previous + W_tilde
    W_tilde is initialized as zero matrix for each new task
    """
    criterion = nn.BCEWithLogitsLoss()
    
    # Store base weights after each task
    base_weights = {}
    
    # Initialize base weights from the model's initial weights
    for name, param in model.named_parameters():
        base_weights[name] = param.data.clone().to(device)
    
    results = []
    
    for task_id, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"Training on {task['name']}")
        print(f"{'='*60}")
        
        # Initialize W_tilde (task-specific weights) as zeros with requires_grad=True
        w_tilde = {}
        for name in base_weights.keys():
            w_tilde[name] = torch.zeros_like(base_weights[name], requires_grad=True, device=device)
        
        # Create optimizer for W_tilde only
        optimizer = torch.optim.SGD(w_tilde.values(), lr=lr)
        
        # Training loop for current task
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(task['train']):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                # Manually set model weights as W = W_base + W_tilde (detached base, trainable tilde)
                for name, param in model.named_parameters():
                    # This creates the computation graph through w_tilde only
                    param.data = (base_weights[name] + w_tilde[name]).detach()
                
                # Now we need to do a custom forward pass where we track w_tilde
                # Let's rebuild the forward computation manually
                x = data
                
                # Layer 1: lin1
                w1 = base_weights['lin1.weight'] + w_tilde['lin1.weight']
                x = F.linear(x, w1)
                x = F.relu(x)
                
                # Layer 2: lin2
                w2 = base_weights['lin2.weight'] + w_tilde['lin2.weight']
                x = F.linear(x, w2)
                x = F.relu(x)
                
                # Layer 3: fc1
                w3 = base_weights['fc1.weight'] + w_tilde['fc1.weight']
                output = F.linear(x, w3)
                
                loss = criterion(output.squeeze(), target.float())
                
                # Backward pass - gradients will flow to w_tilde
                loss.backward()
                
                # Update W_tilde
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                with torch.no_grad():
                    pred = (torch.sigmoid(output.squeeze()) > 0.5).long()
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            
            avg_loss = total_loss / len(task['train'])
            accuracy = 100. * correct / total
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
        
        # Update base weights: W_base = W_base + W_tilde
        with torch.no_grad():
            for name in base_weights.keys():
                base_weights[name] = base_weights[name] + w_tilde[name].data
            
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
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = (torch.sigmoid(output.squeeze()) > 0.5).long()
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    n_hidden = 100
    n_outputs = 1
    epochs = 5
    lr = 0.01
    seed = 0
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load data
    print("Loading Split MNIST dataset...")
    tasks = make_split_mnist(root="./data", train_bs=64, test_bs=512, seed=seed)
    print(f"Number of tasks: {len(tasks)}\n")
    
    # Initialize model
    model = MLPNet(n_hidden=n_hidden, n_outputs=n_outputs).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters\n")
    
    # Train
    start_time = time.time()
    results = train(model, tasks, device, epochs=epochs, lr=lr)
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
        
        # Calculate forgetting (difference between max accuracy and final accuracy)
        forgetting = []
        for task_idx in range(len(tasks) - 1):
            max_acc = max([results[i][task_idx] for i in range(task_idx, len(results))])
            final_acc = results[-1][task_idx]
            forgetting.append(max_acc - final_acc)
        avg_forgetting = sum(forgetting) / len(forgetting) if forgetting else 0
        print(f"Average Forgetting: {avg_forgetting:.2f}%")

if __name__ == "__main__":
    main()