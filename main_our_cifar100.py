import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path
from collections import OrderedDict
import numpy as np
import argparse
from copy import deepcopy
import time

from flatness_minima import SAM
from torch.autograd import Variable

## Define MLP model
class MLPNet(nn.Module):
    def __init__(self, n_hidden=100, n_outputs=10):
        super(MLPNet, self).__init__()
        self.act=OrderedDict()
        self.lin1 = nn.Linear(784,n_hidden,bias=False)
        self.lin2 = nn.Linear(n_hidden,n_hidden, bias=False)
        self.fc1  = nn.Linear(n_hidden, n_outputs, bias=False)
    def forward(self, x):
        self.act['Lin1']=x
        x = self.lin1(x)        
        x = F.relu(x)
        self.act['Lin2']=x
        x = self.lin2(x)        
        x = F.relu(x)
        self.act['fc1']=x
        x = self.fc1(x)
        return x 

# Utils
def get_model(model):
    return deepcopy(model.state_dict())
def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return
def beta_distributions(size, alpha=1):
    return np.random.beta(alpha, alpha, size=size)
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss_a = lam * criterion(pred, y_a)
    loss_b = (1 - lam) * criterion(pred, y_b)
    return loss_a.mean() + loss_b.mean()
class AugModule(nn.Module):
    def __init__(self):
        super(AugModule, self).__init__()
    def forward(self, xs, lam, y, index):
        x_ori = xs
        N = x_ori.size()[0]

        x_ori_perm = x_ori[index, :]

        lam = lam.view((N, 1)).expand_as(x_ori)
        x_mix = (1 - lam) * x_ori + lam * x_ori_perm
        y_a, y_b = y, y[index]
        return x_mix, y_a, y_b

# DEBUG: New test function
def test(args, model, device, x, y, criterion, frozen_weights=None):
    model.eval()
    total_loss = 0
    total_num = 0 
    correct = 0
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r)

    with torch.no_grad():
        # Add frozen weights if provided
        if frozen_weights is not None:
            for name, param in model.named_parameters():
                param.data = frozen_weights[name] + param.data
        
        # Loop batches
        for i in range(0, len(r), args.batch_size_test):
            if i + args.batch_size_test <= len(r): 
                b = r[i:i + args.batch_size_test]
            else: 
                b = r[i:]
            data = x[b].view(-1, 28*28)
            data, target = data.to(device), y[b].to(device)
            output = model(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True) 
            
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num += len(b)
        
        # Restore W_tilde
        if frozen_weights is not None:
            for name, param in model.named_parameters():
                param.data = param.data - frozen_weights[name]

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc

# DEBUG: Split to train_task_0() and train_task_i() with new main()
def train_task_0(args, model, device, x, y, optimizer, criterion):
    """Train the first task (Task 0) without frozen weights"""
    model.train()
    
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r)
    
    for i in range(0, len(r), args.batch_size_train):
        if i + args.batch_size_train <= len(r): 
            b = r[i:i + args.batch_size_train]
        else: 
            b = r[i:]
        
        data = x[b].view(-1, 28*28)
        data, target = data.to(device), y[b].to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


# def train_task_i(args, model, device, x, y, optimizer, criterion, frozen_weights, X_concat=None, eps_1=1e1, U_span_dict=None):
def train_task_i(args, model, device, x, y, optimizer, criterion, frozen_weights, X_concat=None, eps_1=1e1, U_span_dict=None, max_norm=10.0):
    """Train subsequent tasks in the subspace of smaller singular values"""
    
    # ==================== SVD ON CONCATENATED INPUTS ====================
    if X_concat is not None and U_span_dict is None:
        log.info('-'*40)
        log.info('SVD on concatenated inputs from previous tasks')
        log.info('-'*40)
        
        # Center X
        X_mean = torch.mean(X_concat, dim=0, keepdim=True)
        X_centered = X_concat - X_mean
        log.info(f'X_centered shape (before transpose): {X_centered.shape}')
        
        # Transpose X so that m < n (rows < cols)
        X_T = X_centered.T
        log.info(f'X_T shape (after transpose): {X_T.shape}')
        
        # Perform SVD
        U, S, Vt = torch.svd(X_T)
        
        log.info(f'U shape: {U.shape}')
        log.info(f'S shape: {S.shape}')
        log.info(f'Vt shape: {Vt.shape}')
        log.info(f's_j max = {S[0]} - s_j min = {S[-1]}')
        
        # Find maximum singular value s_j that is less than eps_1
        j = None
        s_j = None
        for idx in range(len(S)):
            if S[idx] < eps_1:
                j = idx
                s_j = S[idx].item()
                break
        
        if j is None:
            j = len(S) - 1
            s_j = S[j].item()
            log.info(f'All singular values >= {eps_1}, using last index')
            
        
        # Get U_span: columns from j-th to last column of U
        U_span = U[:, j:]  # Shape: (784, 784-j)
        
        log.info(f'eps_1: {eps_1}')
        log.info(f'j (index of first S < eps_1): {j}')
        log.info(f's_j (singular value at index j): {s_j}')
        log.info(f'X.shape: {X_centered.shape}')
        log.info(f'U_span.shape: {U_span.shape}')
        log.info('-'*40)
        
        # Create U_span_dict
        U_span_dict = {'lin1.weight': U_span.to(device)}
    
    # ==================== TRAINING PHASE ====================
    model.train()
    
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r)
    
    for i in range(0, len(r), args.batch_size_train):
        if i + args.batch_size_train <= len(r): 
            b = r[i:i + args.batch_size_train]
        else: 
            b = r[i:]
        
        data = x[b].view(-1, 28*28)
        data, target = data.to(device), y[b].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: W_effective = W_old + W_tilde @ U_span.T
        # We DON'T change param.data shape - keep gradients aligned
        for name, param in model.named_parameters():
            if name in U_span_dict:
                U_span = U_span_dict[name]
                transformed_W_tilde = torch.mm(param.data, U_span.T)
                # Store original W_tilde temporarily
                # Use .data to avoid breaking computation graph
                original_W_tilde = param.data.clone()
                # Set to W_effective for forward pass
                param.data = frozen_weights[name].to(device) + transformed_W_tilde
            else:
                param.data = frozen_weights[name].to(device) + param.data
        
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Transform gradients back to W_tilde space
        for name, param in model.named_parameters():
            if name in U_span_dict:
                U_span = U_span_dict[name]
                grad_W_tilde = torch.mm(param.grad, U_span)
                
                # Now restore param.data to W_tilde and set the correct gradient
                param.data = param.data - frozen_weights[name].to(device)  # Get transformed W_tilde
                param.data = torch.mm(param.data, U_span)  # Project back to W_tilde space
                
                # Assign the transformed gradient
                param.grad.data = grad_W_tilde
            else:
                param.data = param.data - frozen_weights[name].to(device)
        
        optimizer.step()
        
        # DEBUG: PROJECT W_tilde TO CONSTRAINT SET
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in U_span_dict:
                    U_span = U_span_dict[name]
                    # Compute effective weight
                    W_effective = frozen_weights[name].to(device) + torch.mm(param.data, U_span.T)
                    W_eff_norm = torch.norm(W_effective)
                    
                    # If norm exceeds threshold, scale it back
                    if W_eff_norm > max_norm:
                        scale = max_norm / W_eff_norm
                        W_effective_scaled = W_effective * scale
                        # Back-project to W_tilde space
                        W_tilde_corrected = torch.mm(W_effective_scaled - frozen_weights[name].to(device), U_span)
                        param.data = W_tilde_corrected
                        # log.info(f'Clipped {name}: {W_eff_norm:.4f} -> {max_norm:.4f}')
                else:
                    # For other layers
                    W_effective = frozen_weights[name].to(device) + param.data
                    W_eff_norm = torch.norm(W_effective)
                    
                    if W_eff_norm > max_norm:
                        scale = max_norm / W_eff_norm
                        W_effective_scaled = W_effective * scale
                        param.data = W_effective_scaled - frozen_weights[name].to(device)
                        # log.info(f'Clipped {name}: {W_eff_norm:.4f} -> {max_norm:.4f}')
    
    # Log norms
    log.info('-'*40)
    log.info('Weight Norms after training:')
    for name, param in model.named_parameters():
        W_tilde_norm = torch.norm(param.data).item()
        if name in U_span_dict:
            U_span = U_span_dict[name]
            W_effective = frozen_weights[name].to(device) + torch.mm(param.data, U_span.T)
            W_norm = torch.norm(W_effective).item()
            log.info(f'{name}: ||W|| = {W_norm:.4f}, ||W_tilde|| = {W_tilde_norm:.4f}')
        else:
            W_effective = frozen_weights[name].to(device) + param.data
            W_norm = torch.norm(W_effective).item()
            log.info(f'{name}: ||W|| = {W_norm:.4f}, ||W_tilde|| = {W_tilde_norm:.4f}')
    log.info('-'*40)
    
    return U_span_dict


def main_scratch(args):
    tstart = time.time()
    ## Device Setting 
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ## Load PMNIST DATASET
    from dataloader import pmnist as pmd
    data, taskcla, inputsize = pmd.get(seed=args.seed, pc_valid=args.pc_valid)

    acc_matrix = np.zeros((10, 10))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []
    frozen_weights = None
    X_concat = None
    eps_1 = 1e-1
    
    for k, ncla in taskcla:
        log.info('*'*100)
        log.info(f'Task {k:2d} ({data[k]["name"]:s})')
        log.info('*'*100)
        
        xtrain = data[k]['train']['x']
        ytrain = data[k]['train']['y']
        xvalid = data[k]['valid']['x']
        yvalid = data[k]['valid']['y']
        xtest = data[k]['test']['x']
        ytest = data[k]['test']['y']
        task_list.append(k)

        lr = args.lr 
        log.info('-'*40)
        log.info(f'Task ID :{task_id} | Learning Rate : {lr}')
        log.info('-'*40)
        
        if task_id == 0:
            # ==================== TASK 0 ====================
            model = MLPNet(args.n_hidden, args.n_outputs).to(device)
            log.info('Model parameters ---')
            for k_t, (m, param) in enumerate(model.named_parameters()):
                log.info(f"{k_t}, {m}, {param.shape}")
            log.info('-'*40)

            optimizer = optim.SGD(model.parameters(), lr=lr)

            for epoch in range(1, args.n_epochs+1):
                clock0 = time.time()
                train_task_0(args, model, device, xtrain, ytrain, optimizer, criterion)
                clock1 = time.time()
                
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain, criterion)
                log.info(f'Epoch {epoch:3d} | Train: loss={tr_loss:.3f}, acc={tr_acc:5.1f}% | time={1000*(clock1-clock0):5.1f}ms |')
                
                valid_loss, valid_acc = test(args, model, device, xvalid, yvalid, criterion)
                log.info(f' Valid: loss={valid_loss:.3f}, acc={valid_acc:5.1f}% |')
                log.info('')
            
            log.info('-'*40)
            test_loss, test_acc = test(args, model, device, xtest, ytest, criterion)
            log.info(f'Test: loss={test_loss:.3f} , acc={test_acc:5.1f}%')
            
            frozen_weights = {name: param.data.clone() for name, param in model.named_parameters()}
            log.info('Frozen weights saved after Task 0')
            
            # Log norms for Task 0
            log.info('-'*40)
            log.info('Weight Norms after Task 0:')
            for name, param in model.named_parameters():
                W_norm = torch.norm(param.data).item()
                log.info(f'{name}: ||W|| = {W_norm:.4f}')
            log.info('-'*40)
            
            X_concat = xtrain.view(-1, 28*28).clone()
            log.info(f'Initialized X_concat with Task 0 data, shape: {X_concat.shape}')

        else:
            # ==================== TASK 1+ ====================
            log.info('Re-initializing W_tilde to 0 for new task')
            
            # Compute U_span dimensions from SVD
            log.info('-'*40)
            log.info('Computing U_span dimensions from SVD...')
            
            X_mean = torch.mean(X_concat, dim=0, keepdim=True)
            X_centered = X_concat - X_mean
            X_T = X_centered.T
            U, S, Vt = torch.svd(X_T)
            
            j = None
            for idx in range(len(S)):
                if S[idx] < eps_1:
                    j = idx
                    break
            if j is None:
                j = len(S) - 1
            
            U_span_dim = U.shape[1] - j
            log.info(f'U_span will have {U_span_dim} columns (from index {j} to {U.shape[1]-1})')
            log.info('-'*40)
            
            # Reinitialize model with correct W_tilde dimensions
            model.lin1.weight = nn.Parameter(torch.zeros(100, U_span_dim).to(device))
            model.lin2.weight = nn.Parameter(torch.zeros(100, 100).to(device))
            model.fc1.weight = nn.Parameter(torch.zeros(10, 100).to(device))
            
            log.info('Model parameters (W_tilde) reinitialized ---')
            for k_t, (m, param) in enumerate(model.named_parameters()):
                log.info(f"{k_t}, {m}, {param.shape}")
            log.info('-'*40)
            
            # DEBUG: change optimizer with weight decay (comment the below line)
            # optimizer = optim.SGD(model.parameters(), lr=lr)

            weight_decay = 1e-4  # Tune this: common values are 1e-4, 1e-3, 5e-4
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
            log.info(f'weight decay: {weight_decay}')
            log.info('-'*40)


            # DEBUG: Log norms BEFORE training
            log.info('-'*40)
            log.info('Weight Norms BEFORE training Task {}:'.format(task_id))
            for name, param in model.named_parameters():
                W_tilde_norm = torch.norm(param.data).item()
                if name == 'lin1.weight':
                    # For first layer, also compute effective W
                    U_span_dim = param.shape[1]
                    log.info(f'{name}: ||W_tilde|| = {W_tilde_norm:.4f} (will be transformed with U_span)')
                else:
                    log.info(f'{name}: ||W_tilde|| = {W_tilde_norm:.4f}')
            log.info('-'*40)
            
            # Training loop
            U_span_dict = None
            for epoch in range(1, args.n_epochs+1):
                clock0 = time.time()
                
                # if epoch == 1:
                #     U_span_dict = train_task_i(args, model, device, xtrain, ytrain, optimizer, criterion, frozen_weights, X_concat, eps_1, U_span_dict)
                # else:
                #     train_task_i(args, model, device, xtrain, ytrain, optimizer, criterion, frozen_weights, None, eps_1, U_span_dict)

                if epoch == 1:
                    U_span_dict = train_task_i(args, model, device, xtrain, ytrain, optimizer, criterion, frozen_weights, X_concat, eps_1, U_span_dict, max_norm=10.0)
                else:
                    train_task_i(args, model, device, xtrain, ytrain, optimizer, criterion, frozen_weights, None, eps_1, U_span_dict, max_norm=10.0)
                
                clock1 = time.time()
                
                # DEBUG: Log norm
                log.info('-'*40)
                log.info('Weight Norms AFTER Epoch {}:'.format(epoch))
                for name, param in model.named_parameters():
                    W_tilde_norm = torch.norm(param.data).item()
                    if name in U_span_dict:
                        U_span = U_span_dict[name]
                        W_effective = frozen_weights[name].to(device) + torch.mm(param.data, U_span.T)
                        W_norm = torch.norm(W_effective).item()
                        log.info(f'{name}: ||W_effective|| = {W_norm:.4f}, ||W_tilde|| = {W_tilde_norm:.4f}')
                    else:
                        W_effective = frozen_weights[name].to(device) + param.data
                        W_norm = torch.norm(W_effective).item()
                        log.info(f'{name}: ||W_effective|| = {W_norm:.4f}, ||W_tilde|| = {W_tilde_norm:.4f}')
                log.info('-'*40)
                
                # For evaluation, need to use frozen_weights + W_tilde @ U_span.T
                # Temporarily set weights for evaluation
                original_params = {}
                for name, param in model.named_parameters():
                    original_params[name] = param.data.clone()
                    if name in U_span_dict:
                        U_span = U_span_dict[name]
                        transformed = torch.mm(param.data, U_span.T)
                        param.data = frozen_weights[name].to(device) + transformed
                    else:
                        param.data = frozen_weights[name].to(device) + param.data
                
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain, criterion)
                log.info(f'Epoch {epoch:3d} | Train: loss={tr_loss:.3f}, acc={tr_acc:5.1f}% | time={1000*(clock1-clock0):5.1f}ms |')
                
                valid_loss, valid_acc = test(args, model, device, xvalid, yvalid, criterion)
                log.info(f' Valid: loss={valid_loss:.3f}, acc={valid_acc:5.1f}% |')
                log.info('')
                
                # Restore W_tilde
                for name, param in model.named_parameters():
                    param.data = original_params[name]
            
            # Final test
            original_params = {}
            for name, param in model.named_parameters():
                original_params[name] = param.data.clone()
                if name in U_span_dict:
                    U_span = U_span_dict[name]
                    transformed = torch.mm(param.data, U_span.T)
                    param.data = frozen_weights[name].to(device) + transformed
                else:
                    param.data = frozen_weights[name].to(device) + param.data
            
            test_loss, test_acc = test(args, model, device, xtest, ytest, criterion)
            log.info(f'Test: loss={test_loss:.3f} , acc={test_acc:5.1f}%')
            
            # Restore and update frozen weights
            for name, param in model.named_parameters():
                param.data = original_params[name]
            
            log.info('Updating frozen weights: W_frozen = W_frozen + U_span @ W_tilde')
            for name, param in model.named_parameters():
                if name in U_span_dict:
                    U_span = U_span_dict[name]
                    transformed_W_tilde = torch.mm(param.data, U_span.T)
                    frozen_weights[name] = frozen_weights[name].to(device) + transformed_W_tilde
                else:
                    frozen_weights[name] = frozen_weights[name].to(device) + param.data.clone()
            log.info('-'*40)
            
            X_current = xtrain.view(-1, 28*28)
            X_concat = torch.cat([X_concat, X_current], dim=0)
            log.info(f'Added Task {task_id} data to X_concat, new shape: {X_concat.shape}')
            log.info('-'*40)

        # ==================== EVALUATE ALL TASKS ====================
        jj = 0 
        for ii in np.array(task_list)[0:task_id+1]:
            xtest_eval = data[ii]['test']['x']
            ytest_eval = data[ii]['test']['y']
            
            if task_id == 0:
                _, acc_matrix[task_id, jj] = test(args, model, device, xtest_eval, ytest_eval, criterion)
            else:
                original_params = {}
                for name, param in model.named_parameters():
                    original_params[name] = param.data.clone()
                    param.data = frozen_weights[name].clone()
                
                _, acc_matrix[task_id, jj] = test(args, model, device, xtest_eval, ytest_eval, criterion)
                
                for name, param in model.named_parameters():
                    param.data = original_params[name]
            
            jj += 1
        
        log.info('Accuracies =')
        for i_a in range(task_id + 1):
            acc_ = ''
            for j_a in range(acc_matrix.shape[1]):
                acc_ += f'{acc_matrix[i_a, j_a]:5.1f}% '
            log.info(acc_)

        task_id += 1

    log.info('-'*50)
    log.info(f'Task Order : {np.array(task_list)}')
    log.info(f'Final Avg Accuracy: {acc_matrix[-1].mean():5.2f}%')
    bwt = np.mean((acc_matrix[-1] - np.diag(acc_matrix))[:-1]) 
    log.info(f'Backward transfer: {bwt:5.2f}%')
    log.info(f'[Elapsed time = {(time.time()-tstart)*1000:.1f} ms]')
    log.info('-'*50)
    
    return acc_matrix[-1].mean(), bwt

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with DFGP')
    parser.add_argument('--batch_size_train', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    # DEBUG using n_epochs=3 instead of 5
    parser.add_argument('--n_epochs', type=int, default=5, metavar='N',
                        help='number of training epochs/task (default: 5)')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 2)')
    parser.add_argument('--pc_valid',default=0.1,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    # Architecture
    parser.add_argument('--n_hidden', type=int, default=100, metavar='NH',
                        help='number of hidden units in MLP (default: 100)')
    parser.add_argument('--n_outputs', type=int, default=10, metavar='NO',
                        help='number of output units in MLP (default: 10)')
    parser.add_argument('--n_tasks', type=int, default=10, metavar='NT',
                        help='number of tasks (default: 10)')
    parser.add_argument('--savename', type=str, default='./logs/PMNIST/',
                        help='save path')
    parser.add_argument('--gpm_thro1', type=float, default=0.95, metavar='THR1',
                        help='projection thr1')
    parser.add_argument('--gpm_thro2', type=float, default=0.99, metavar='THR2',
                        help='projection thr2')
    parser.add_argument('--gpm_thro3', type=float, default=0.99, metavar='THR3',
                        help='projection thr3')
    parser.add_argument('--mixup_alpha', type=float, default=20, metavar='Alpha',
                        help='mixup_alpha')
    parser.add_argument('--mixup_weight', type=float, default=0.1, metavar='Weight',
                        help='mixup_weight')

    # parser.add_argument('--max_norm', type=float, default=10.0, metavar='MN',
    #                 help='maximum norm for effective weights (default: 10.0)')

    args = parser.parse_args()
    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    # log = create_log_dir(args.savename, 'log_{}.txt'.format(str_time_))
    log = create_log_dir(args.savename, f'log_{str_time_}.txt')

    # for thro_1 in [0.94, 0.95, 0.96]:
    #     for thro_2_and_3 in [0.96, 0.97, 0.98, 0.99]:
    #         for mixup_weight in [0.01, 0.001, 0.0001]:

    for thro_1 in [0.96]:
        for thro_2_and_3 in [0.99]:
            for mixup_weight in [0.01]:

                accs, bwts = [], []
                str_time = str_time_ + '_' + str(thro_1) + '_' + str(thro_2_and_3) + '_' + str(thro_2_and_3)

                args.mixup_weight = mixup_weight
                args.gpm_thro1 = thro_1
                args.gpm_thro2 = thro_2_and_3
                args.gpm_thro3 = thro_2_and_3

                # for seed_ in [1, 2]:
                for seed_ in [1]:
                    try:
                        args.seed = seed_
                        log.info('=' * 100)
                        log.info('Arguments =')
                        log.info(str(args))
                        log.info('=' * 100)

                        # acc, bwt = main_scratch(args)
                        acc, bwt = main_scratch(args)
                        accs.append(acc)
                        bwts.append(bwt)
                        
                    except Exception as e:
                        log.error(f"seed {seed_} Error: {type(e).__name__}: {str(e)}")
                        import traceback
                        log.error(traceback.format_exc())



