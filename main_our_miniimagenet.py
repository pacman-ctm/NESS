import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from torch.autograd import Variable
import os
import os.path
from collections import OrderedDict
import numpy as np
import argparse,time
from copy import deepcopy
import time
from flatness_minima import SAM

## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            )
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x):
        self.count = self.count % 2 
         # self.act['conv_{}'.format(self.count)] = x
        self.act[f"conv_{self.count}"] = x
        self.count +=1
        out = relu(self.bn1(self.conv1(x)))
        self.count = self.count % 2 
        # self.act['conv_{}'.format(self.count)] = out
        self.act[f"conv_{self.count}"] = out
        self.count +=1
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, taskcla, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1, 2)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        
        self.taskcla = taskcla
        self.linear=torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.linear.append(nn.Linear(nf * 8 * block.expansion * 9, n, bias=False))
        self.act = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        bsz = x.size(0)
        self.act['conv_in'] = x.view(bsz, 3, 84, 84)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 84, 84))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        y=[]
        for t,i in self.taskcla:
            y.append(self.linear[t](out))
        return y
def ResNet18(taskcla, nf=32):
    return ResNet(BasicBlock, [2, 2, 2, 2], taskcla, nf)
def get_model(model):
    return deepcopy(model.state_dict())
def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return
def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        if (epoch ==1):
            param_group['lr']=args.lr
        else:
            param_group['lr'] /= args.lr_factor  
def beta_distributions(size, alpha=1):
    return np.random.beta(alpha, alpha, size=size)
class AugModule(nn.Module):
    def __init__(self):
        super(AugModule, self).__init__()
    def forward(self, xs, lam, y, index):
        x_ori = xs
        N = x_ori.size()[0]
        x_ori_perm = x_ori[index, :]
        lam = lam.view((N, 1, 1, 1)).expand_as(x_ori)
        x_mix = (1 - lam) * x_ori + lam * x_ori_perm
        y_a, y_b = y, y[index]
        return x_mix, y_a, y_b
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss_a = lam * criterion(pred, y_a)
    loss_b = (1 - lam) * criterion(pred, y_b)
    return loss_a.mean() + loss_b.mean()

def test(args, model, device, x, y, criterion, task_id):
    model.eval()
    total_loss = 0
    total_num = 0 
    correct = 0
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    # r=torch.LongTensor(r).to(device)
    r = torch.LongTensor(r) # DEBUG

    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_test):
            if i+args.batch_size_test<=len(r): b=r[i:i+args.batch_size_test]
            else: b=r[i:]
            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output = model(data)
            loss = criterion(output[task_id], target)
            pred = output[task_id].argmax(dim=1, keepdim=True) 
            
            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc

def train_task_0(args, model, device, x, y, optimizer, criterion, task_id):
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
        
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        
        optimizer.zero_grad()
        output = model(data)[task_id]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def train_task_i(args, model, device, x, y, optimizer, criterion, frozen_weights, task_id, X_concat=None, eps_1=1e1, U_span_dict=None):
    """Train subsequent tasks in the subspace of smaller singular values"""
    
    # ==================== SVD ON CONCATENATED INPUTS ====================
    if X_concat is not None and U_span_dict is None:
        log.info('-'*40)
        log.info('SVD on concatenated inputs from previous tasks')
        log.info('-'*40)
        
        X_mean = torch.mean(X_concat, dim=0, keepdim=True)
        X_centered = X_concat - X_mean
        X_T = X_centered.T
        
        U, S, Vt = torch.svd(X_T)
        
        log.info(f'U shape: {U.shape}')
        log.info(f'S shape: {S.shape}')
        log.info(f'Vt shape: {Vt.shape}')
        log.info(f's_j max = {S[0]} - s_j min = {S[-1]}')
        
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
        
        U_span = U[:, j:]
        
        log.info(f'eps_1: {eps_1}')
        log.info(f'j (index of first S < eps_1): {j}')
        log.info(f's_j (singular value at index j): {s_j}')
        log.info(f'X.shape: {X_centered.shape}')
        log.info(f'U_span.shape: {U_span.shape}')
        log.info('-'*40)
        
        # Store U_span ONLY for the current task's linear layer
        U_span_dict = {f'linear.{task_id}.weight': U_span.to(device)}
    
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
        
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: Set effective weights
        # Store original W_tilde before transformation
        original_W_tilde = {}
        for name, param in model.named_parameters():
            if param.requires_grad:  # Only for trainable parameters
                original_W_tilde[name] = param.data.clone()
                
                if name in U_span_dict:
                    # Current task's linear layer: W_effective = W_frozen + W_tilde @ U_span.T
                    U_span = U_span_dict[name]
                    transformed_W_tilde = torch.mm(param.data, U_span.T)
                    param.data = frozen_weights[name].to(device) + transformed_W_tilde
                elif name in frozen_weights:
                    # Other trainable layers: W_effective = W_frozen + W_tilde
                    param.data = frozen_weights[name].to(device) + param.data
        
        output = model(data)[task_id]
        loss = criterion(output, target)
        loss.backward()
        
        # Restore W_tilde and transform gradients
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
                
            if name in U_span_dict:
                # Transform gradient: grad_W_tilde = grad_W_effective @ U_span
                U_span = U_span_dict[name]
                grad_W_tilde = torch.mm(param.grad, U_span)
                
                # Restore original W_tilde
                param.data = original_W_tilde[name]
                
                # Set transformed gradient
                param.grad.data = grad_W_tilde
            elif name in original_W_tilde:
                # Restore original W_tilde
                param.data = original_W_tilde[name]
        
        optimizer.step()
    
    return U_span_dict


def main_scratch(args):
    tstart = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from dataloader import miniimagenet as data_loader
    dataloader = data_loader.DatasetGen(args)
    taskcla, inputsize = dataloader.taskcla, dataloader.inputsize

    acc_matrix = np.zeros((10, 10))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []
    frozen_weights = None
    X_concat = None
    eps_1 = 1e1
    
    for k, ncla in taskcla:
        data = dataloader.get(k)
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
            model = ResNet18(taskcla, nf=32).to(device)
            log.info('Model parameters ---')
            for k_t, (m, param) in enumerate(model.named_parameters()):
                log.info(f"{k_t}, {m}, {param.shape}")
            log.info('-'*40)

            optimizer = optim.SGD(model.parameters(), lr=lr)

            for epoch in range(1, args.n_epochs+1):
                clock0 = time.time()
                train_task_0(args, model, device, xtrain, ytrain, optimizer, criterion, k)
                clock1 = time.time()
                
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain, criterion, k)
                log.info(f'Epoch {epoch:3d} | Train: loss={tr_loss:.3f}, acc={tr_acc:5.1f}% | time={1000*(clock1-clock0):5.1f}ms |')
                
                valid_loss, valid_acc = test(args, model, device, xvalid, yvalid, criterion, k)
                log.info(f' Valid: loss={valid_loss:.3f}, acc={valid_acc:5.1f}% |')
                log.info('')
            
            log.info('-'*40)
            test_loss, test_acc = test(args, model, device, xtest, ytest, criterion, k)
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

            # For ResNet, we need to extract the flattened features before linear layers
            with torch.no_grad():
                model.eval()
                batch_size = 64
                features_list = []
                for i in range(0, xtrain.size(0), batch_size):
                    batch = xtrain[i:i+batch_size].to(device)
                    bsz = batch.size(0)
                    
                    # Forward pass through the network
                    out = relu(model.bn1(model.conv1(batch.view(bsz, 3, 84, 84))))
                    out = model.layer1(out)
                    out = model.layer2(out)
                    out = model.layer3(out)
                    out = model.layer4(out)
                    out = avg_pool2d(out, 2)
                    out = out.view(out.size(0), -1)  # This is the feature before linear layer
                    
                    features_list.append(out.cpu().clone())
                
                X_concat = torch.cat(features_list, dim=0)
                log.info(f'Initialized X_concat with Task 0 features (before linear), shape: {X_concat.shape}')

        else:
            # ==================== TASK 1+ ====================
            log.info('Re-initializing W_tilde to 0 for new task')
            
            # Compute U_span dimensions from SVD
            log.info('-'*40)
            log.info('Computing U_span dimensions from SVD...')
            
            with torch.no_grad():
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
            
            # Store the original full-dimensional frozen weight for current task
            linear_layer_name = f'linear.{k}.weight'
            original_full_dim_weight = frozen_weights[linear_layer_name].clone()
            
            # Reinitialize ONLY the current task's linear layer with reduced dimensions
            model.linear[k].weight = nn.Parameter(torch.zeros(ncla, U_span_dim).to(device))
            
            # Update frozen_weights for the current task to the full dimension
            frozen_weights[linear_layer_name] = original_full_dim_weight.to(device)
            
            # Set all previous tasks' linear layers to frozen (not trainable)
            for prev_task_id in range(task_id):
                prev_linear_name = f'linear.{prev_task_id}.weight'
                if prev_linear_name in frozen_weights:
                    # Set to frozen weight and disable gradients
                    model.linear[prev_task_id].weight.data = frozen_weights[prev_linear_name].to(device).clone()
                    model.linear[prev_task_id].weight.requires_grad = False
            
            # Reset all trainable layers (conv, BN) to zero (W_tilde = 0)
            for name, param in model.named_parameters():
                if param.requires_grad and 'linear' not in name:
                    param.data.zero_()
            
            log.info('Model parameters (W_tilde) reinitialized ---')
            for k_t, (m, param) in enumerate(model.named_parameters()):
                log.info(f"{k_t}, {m}, {param.shape}, requires_grad={param.requires_grad}")
            log.info('-'*40)
            
            weight_decay = 1e-4
            # Create optimizer with only trainable parameters
            trainable_params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    trainable_params.append(param)
            
            optimizer = optim.SGD(trainable_params, lr=lr, weight_decay=weight_decay)
            log.info(f'weight decay: {weight_decay}')
            log.info(f'Number of trainable parameters: {sum(p.numel() for p in trainable_params)}')
            log.info('-'*40)

            # Log norms BEFORE training
            log.info('-'*40)
            log.info(f'Weight Norms BEFORE training Task {task_id}:')
            for name, param in model.named_parameters():
                W_tilde_norm = torch.norm(param.data).item()
                log.info(f'{name}: ||W_tilde|| = {W_tilde_norm:.4f}, requires_grad={param.requires_grad}')
            log.info('-'*40)
            
            # Training loop
            U_span_dict = None
            for epoch in range(1, args.n_epochs+1):
                clock0 = time.time()
                
                if epoch == 1:
                    U_span_dict = train_task_i(args, model, device, xtrain, ytrain, optimizer, criterion, frozen_weights, k, X_concat, eps_1, U_span_dict)
                else:
                    train_task_i(args, model, device, xtrain, ytrain, optimizer, criterion, frozen_weights, k, None, eps_1, U_span_dict)
                
                clock1 = time.time()
                
                # Log norms after epoch
                if epoch % 50 == 0 or epoch == 1:
                    log.info('-'*40)
                    log.info(f'Weight Norms AFTER Epoch {epoch}:')
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            W_tilde_norm = torch.norm(param.data).item()
                            if name in U_span_dict:
                                U_span = U_span_dict[name]
                                W_effective = frozen_weights[name].to(device) + torch.mm(param.data, U_span.T)
                                W_norm = torch.norm(W_effective).item()
                                log.info(f'{name}: ||W_effective|| = {W_norm:.4f}, ||W_tilde|| = {W_tilde_norm:.4f}')
                            elif name in frozen_weights:
                                W_effective = frozen_weights[name].to(device) + param.data
                                W_norm = torch.norm(W_effective).item()
                                log.info(f'{name}: ||W_effective|| = {W_norm:.4f}, ||W_tilde|| = {W_tilde_norm:.4f}')
                    log.info('-'*40)
                
                # For evaluation, temporarily set weights
                original_params = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        original_params[name] = param.data.clone()
                        if name in U_span_dict:
                            U_span = U_span_dict[name]
                            transformed = torch.mm(param.data, U_span.T)
                            param.data = frozen_weights[name].to(device) + transformed
                        elif name in frozen_weights:
                            param.data = frozen_weights[name].to(device) + param.data
                
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain, criterion, k)
                log.info(f'Epoch {epoch:3d} | Train: loss={tr_loss:.3f}, acc={tr_acc:5.1f}% | time={1000*(clock1-clock0):5.1f}ms |')
                
                valid_loss, valid_acc = test(args, model, device, xvalid, yvalid, criterion, k)
                log.info(f' Valid: loss={valid_loss:.3f}, acc={valid_acc:5.1f}% |')
                
                # Restore W_tilde
                for name, param in model.named_parameters():
                    if name in original_params:
                        param.data = original_params[name]
            
            # Final test
            original_params = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    original_params[name] = param.data.clone()
                    if name in U_span_dict:
                        U_span = U_span_dict[name]
                        transformed = torch.mm(param.data, U_span.T)
                        param.data = frozen_weights[name].to(device) + transformed
                    elif name in frozen_weights:
                        param.data = frozen_weights[name].to(device) + param.data
            
            test_loss, test_acc = test(args, model, device, xtest, ytest, criterion, k)
            log.info(f'Test: loss={test_loss:.3f} , acc={test_acc:5.1f}%')
            
            # Restore W_tilde
            for name, param in model.named_parameters():
                if name in original_params:
                    param.data = original_params[name]
            
            log.info('Updating frozen weights: W_frozen = W_frozen + W_tilde @ U_span.T or W_tilde')
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in U_span_dict:
                        U_span = U_span_dict[name]
                        transformed_W_tilde = torch.mm(param.data, U_span.T)
                        frozen_weights[name] = frozen_weights[name].to(device) + transformed_W_tilde
                    elif name in frozen_weights:
                        frozen_weights[name] = frozen_weights[name].to(device) + param.data.clone()
            log.info('-'*40)

            # Extract features from current task
            with torch.no_grad():
                model.eval()
                # Set all weights to frozen for feature extraction
                original_params = {}
                for name, param in model.named_parameters():
                    original_params[name] = param.data.clone()
                    if name in frozen_weights:
                        param.data = frozen_weights[name].to(device).clone()
                
                batch_size = 64
                features_list = []
                for i in range(0, xtrain.size(0), batch_size):
                    batch = xtrain[i:i+batch_size].to(device)
                    bsz = batch.size(0)
                    
                    # Forward pass
                    out = relu(model.bn1(model.conv1(batch.view(bsz, 3, 84, 84))))
                    out = model.layer1(out)
                    out = model.layer2(out)
                    out = model.layer3(out)
                    out = model.layer4(out)
                    out = avg_pool2d(out, 2)
                    out = out.view(out.size(0), -1)
                    
                    features_list.append(out.cpu().clone())
                
                X_current = torch.cat(features_list, dim=0)
                
                # Restore parameters
                for name, param in model.named_parameters():
                    param.data = original_params[name]
                
            X_concat = torch.cat([X_concat, X_current], dim=0)
            log.info(f'Added Task {task_id} features to X_concat, new shape: {X_concat.shape}')

        # ==================== EVALUATE ALL TASKS ====================
        jj = 0 
        for ii in np.array(task_list)[0:task_id+1]:
            data_eval = dataloader.get(ii)  # Fixed: Need to get data for each task
            xtest_eval = data_eval[ii]['test']['x']
            ytest_eval = data_eval[ii]['test']['y']
            
            if task_id == 0:
                _, acc_matrix[task_id, jj] = test(args, model, device, xtest_eval, ytest_eval, criterion, ii)
            else:
                original_params = {}
                for name, param in model.named_parameters():
                    original_params[name] = param.data.clone()
                    param.data = frozen_weights[name].clone()
                
                _, acc_matrix[task_id, jj] = test(args, model, device, xtest_eval, ytest_eval, criterion, ii)
                
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
    parser = argparse.ArgumentParser(description='miniimagenet datasets with DFGP')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=2, metavar='N',
                        help='number of training epochs/task (default: 100)')
    parser.add_argument('--seed', type=int, default=37, metavar='S',
                        help='random seed (default: 37)')
    parser.add_argument('--pc_valid',default=0.02,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-3, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=5, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=3, metavar='LRF',
                        help='lr decay factor (default: 2)')
    parser.add_argument('--savename', type=str, default='./logs/MINI/',
                        help='save path')
    parser.add_argument('--gpm_thro', type=float, default=0.97, metavar='gradient projection',
                        help='gpm_thro')
    parser.add_argument('--mixup_alpha', type=float, default=20, metavar='Alpha',
                        help='mixup_alpha')
    parser.add_argument('--mixup_weight', type=float, default=0.1, metavar='Weight',
                        help='mixup_weight')

    args = parser.parse_args()

    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    # log = create_log_dir(args.savename, 'log_{}.txt'.format(str_time_))
    log = create_log_dir(args.savename, f'log_{str_time_}.txt')

    # for mixup_weight in [0.0001, 0.01, 0.05, 0.1]:
    for mixup_weight in [0.0001]:

        accs, bwts = [], []
        str_time = str_time_ + '_' + str(mixup_weight)
        args.mixup_weight = mixup_weight

        # for seed_ in [1, 2]:
        for seed_ in [1]:
            try:
                args.seed = seed_
                log.info('=' * 100)
                log.info('Arguments =')
                log.info(str(args))
                log.info('=' * 100)

                # acc, bwt = main(args)
                acc, bwt = main_scratch(args)
                accs.append(acc)
                bwts.append(bwt)
            except Exception as e:
                log.error(f"seed {seed_} Error: {type(e).__name__}: {str(e)}")
                import traceback
                log.error(traceback.format_exc())




