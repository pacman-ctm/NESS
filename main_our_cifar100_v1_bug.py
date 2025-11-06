import torch
import torch.optim as optim
import torch.nn as nn

from torch.autograd import Variable
import os
import os.path
from collections import OrderedDict
import numpy as np
import argparse,time
from copy import deepcopy
import time

import torch.nn.functional as F

## Define AlexNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class AlexNet(nn.Module):
    def __init__(self,taskcla):
        super(AlexNet, self).__init__()
        self.act=OrderedDict()
        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        s=compute_conv_output_size(32,4)
        s=s//2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        s=compute_conv_output_size(s,3)
        s=s//2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.ksize.append(2)
        self.in_channel.append(128)
        self.map.append(256*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256*self.smid*self.smid,2048, bias=False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.fc2 = nn.Linear(2048,2048, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.map.extend([2048])
        
        self.taskcla = taskcla
        self.fc3=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.fc3.append(torch.nn.Linear(2048,n,bias=False))
    def forward(self, x):
        bsz = deepcopy(x.size(0))
        self.act['conv1']=x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        self.act['conv2']=x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        self.act['conv3']=x
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x=x.view(bsz,-1)
        self.act['fc1']=x
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        self.act['fc2']=x        
        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))

        y=[]
        for t,i in self.taskcla:
            y.append(self.fc3[t](x))
            
        return y

# DEBUG: Adapt code


def im2col(input_tensor, kernel_size, stride, device):
    """
    Converts the input tensor into columns.

    Args:
    input_tensor (torch.Tensor): Input tensor of shape (
        batch_size, in_channels, height, width).
    kernel_size (tuple): The size of the convolution kernel (
        kernel_height, kernel_width).
    stride (tuple): The stride of the convolution (
        stride_height, stride_width).
    device (str): Device to move the tensor to.

    Returns:
    torch.Tensor: Column matrix.
    """
    batch_size, in_channels, height, width = input_tensor.shape
    kernel_height, kernel_width = kernel_size
    stride_height, stride_width = stride

    # Calculate output dimensions
    out_height = (height - kernel_height) // stride_height + 1
    out_width = (width - kernel_width) // stride_width + 1

    col = torch.zeros(
        size=(batch_size, in_channels,
              kernel_height, kernel_width,
              out_height, out_width)).to(device)

    for y in range(kernel_height):
        y_max = y + stride_height * out_height
        for x in range(kernel_width):
            x_max = x + stride_width * out_width
            col[:, :, y, x, :, :] = input_tensor[
                :, :, y:y_max:stride_height, x:x_max:stride_width]

    col = col.permute(0, 4, 5, 1, 2, 3).contiguous()
    col = col.view(batch_size * out_height * out_width, -1)
    return col


def get_inputs(network, layer, data_loader, max_data_count, device,
               batchwise_transform=torch.nn.Identity()):
    # record layer activations
    inputs = None
    data_count = 0

    if isinstance(layer, nn.Conv2d):
        assert layer.dilation == (1, 1), "Dilation not supported"
        assert layer.groups == 1, "Groups not supported"

    def hook(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        nonlocal inputs
        input_matrix = input[0]

        # if layer is conv2d preprocess input
        if isinstance(layer, nn.Conv2d):
            # Padding inputs
            input_matrix = F.pad(
                input_matrix, (layer.padding[1], layer.padding[1],
                               layer.padding[0], layer.padding[0]))
            # Im2col Images to Columns for replacing convolution
            # with matrix multiplication
            input_matrix = im2col(
                input_matrix, layer.kernel_size, layer.stride, device)

        # if input is from LLM reshape
        if input_matrix.dim() == 3:
            input_matrix = input_matrix.reshape(-1, input_matrix.shape[-1])

        if inputs is None:
            inputs = input_matrix.T @ input_matrix
        else:
            inputs += input_matrix.T @ input_matrix

    handle = layer.register_forward_hook(hook)

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            data = batchwise_transform(data)
            network(data)
            data_count += data.size(0)
            if data_count >= max_data_count:
                break
    handle.remove()
    return inputs

def replace_layer_by_name(model, layer_name, new_layer):
    def recursive_replace(module, names):
        if len(names) == 1:
            if hasattr(module, names[0]):
                setattr(module, names[0], new_layer)
                return True
            else:
                return False
        else:
            name = names[0]
            if name.isdigit():
                name = int(name)
                if isinstance(module, (nn.Sequential, nn.ModuleList)):
                    if name < len(module):
                        return recursive_replace(module[name], names[1:])
                    else:
                        return False
            else:
                child = getattr(module, name, None)
                if child is None:
                    return False
                return recursive_replace(child, names[1:])

    names = layer_name.split('.')
    if not recursive_replace(model, names):
        raise ValueError(f"Layer '{layer_name}' not found in the model.")


class LinearAdapt(nn.Module):
    def __init__(self, old_linear: nn.Linear, cross_inputs: torch.Tensor, eps_rel_tol: float = 1e-2):
        super(LinearAdapt, self).__init__()
        self.old_linear = old_linear
        # old linear weight is not trainable
        self.old_linear.weight.requires_grad = False
        if self.old_linear.bias is not None:
            self.old_linear.bias.requires_grad = False
        self.device = self.old_linear.weight.device
        # get U and V matrices from cross inputs
        self._get_UV(cross_inputs, eps_rel_tol)

    def _get_UV(self, inputs: torch.Tensor, eps_rel_tol: float):
        S_X_squared, V_X = torch.linalg.eigh(inputs)
        S_X_squared.clamp(min=0.0)  # in-place clamping since matrix is psd
        V_Xt = V_X.T 
        S_X = torch.sqrt(S_X_squared)
        print(f"Singular values: {S_X}")

        eps_tol = eps_rel_tol * S_X.sum()
        print(f"{eps_rel_tol = } - {eps_tol = }")
        zero_rank = torch.sum(S_X <= eps_tol).item()
        # DEBUG: test truncated
        # zero_rank = 5

        if zero_rank == 0:

            self.weight_U = None
            self.weight_V = None
        else:
            # print(f"V_Xt before truncate = {V_Xt} - shape = {V_Xt.shape}")
            V_Xt = V_Xt[:zero_rank, :]
            # V_Xt = V_Xt[-zero_rank:, :]
            # print(f"V_Xt after truncate = {V_Xt} - shape = {V_Xt.shape}")
            # not trainable parameter
            self.weight_V = nn.Parameter(V_Xt.T, requires_grad=False)
            # trainable parameter initialized to zero
            self.weight_U = nn.Parameter(torch.zeros(
                self.old_linear.out_features, zero_rank).to(self.device),
                                         requires_grad=True)
        print(f"Adaptation rank: {self.rank}")

    @property
    def rank(self):
        if self.weight_V is None:
            return 0
        return self.weight_V.shape[1]

    @property
    def delta_W(self):
        if self.weight_V is None:
            return torch.zeros_like(self.old_linear.weight)
        return self.weight_U @ self.weight_V.T

    def reset_parameters(self):
        if self.weight_U is not None:
            nn.init.zeros_(self.weight_U)

    def forward(self, input):
        return F.linear(input, self.old_linear.weight + self.delta_W, self.old_linear.bias)


class Conv2dAdapt(nn.Module):
    def __init__(self, old_conv: nn.Conv2d, cross_inputs: torch.Tensor, eps_rel_tol: float = 1e-2):
        super(Conv2dAdapt, self).__init__()
        self.old_conv = old_conv
        # old conv weight is not trainable
        self.old_conv.weight.requires_grad = False
        if self.old_conv.bias is not None:
            self.old_conv.bias.requires_grad = False
        self.device = self.old_conv.weight.device
        # get U and V matrices from cross inputs
        self._get_UV(cross_inputs, eps_rel_tol)

    def _get_UV(self, inputs: torch.Tensor, eps_rel_tol: float):
        S_X_squared, V_X = torch.linalg.eigh(inputs)
        S_X_squared.clamp(min=0.0)  # in-place clamping since matrix is psd
        V_Xt = V_X.T
        S_X = torch.sqrt(S_X_squared)
        print(f"Singular values: {S_X}")

        eps_tol = eps_rel_tol * S_X.sum()
        print(f"{S_X.sum() = }")
        print(f"{eps_rel_tol = } - {eps_tol = }")
        zero_rank = torch.sum(S_X <= eps_tol).item()
        # DEBUG: test truncated
        # zero_rank = 10

        if zero_rank == 0:
            self.weight_U = None
            self.weight_V = None
        else:
            # print(f"V_Xt before truncate = {V_Xt} - shape = {V_Xt.shape}")
            V_Xt = V_Xt[:zero_rank, :]
            # print(f"V_Xt after truncate = {V_Xt} - shape = {V_Xt.shape}")
            # not trainable parameter
            self.weight_V = nn.Parameter(V_Xt.T, requires_grad=False)
            # trainable parameter initialized to zero
            self.weight_U = nn.Parameter(torch.zeros(
                self.old_conv.out_channels, zero_rank).to(self.device),
                                         requires_grad=True)
        print(f"Adaptation rank: {self.rank}")

    @property
    def rank(self):
        if self.weight_V is None:
            return 0
        return self.weight_V.shape[1]

    @property
    def delta_W(self):
        if self.weight_V is None:
            return torch.zeros_like(self.old_conv.weight)
        return self.weight_U @ self.weight_V.T

    def reset_parameters(self):
        if self.weight_U is not None:
            nn.init.zeros_(self.weight_U)

    def forward(self, input):
        delta_W_reshaped = self.delta_W.view_as(self.old_conv.weight)
        return F.conv2d(input, self.old_conv.weight + delta_W_reshaped,
                        self.old_conv.bias, stride=self.old_conv.stride,
                        padding=self.old_conv.padding,
                        dilation=self.old_conv.dilation,
                        groups=self.old_conv.groups)

# END DEBUG


# DEBUG: Train, test and main functions

def train_epoch(model, train_loader, optimizer, criterion, device, task_id):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        
        # Select output head for current task
        loss = criterion(outputs[task_id], target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = outputs[task_id].argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_first_task(model, task_data, args, device, log):
    """Train all weights from scratch for first task"""
    log.info("Training first task from scratch...")
    
    # Create dataloader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(task_data['train']['x'], 
                                       task_data['train']['y']),
        batch_size=args.batch_size_train, shuffle=True)
    
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(task_data['valid']['x'], 
                                       task_data['valid']['y']),
        batch_size=args.batch_size_test, shuffle=False)
    
    # Optimizer for all parameters
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_valid_acc = 0
    patience_counter = 0
    
    for epoch in range(args.n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, task_id=0)
        
        # Validation
        model.eval()
        valid_correct = 0
        valid_total = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                pred = outputs[0].argmax(dim=1)
                valid_correct += (pred == target).sum().item()
                valid_total += target.size(0)
        
        valid_acc = 100. * valid_correct / valid_total
        
        log.info(f'Epoch {epoch+1}/{args.n_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Valid Acc: {valid_acc:.2f}%')
        
        # Learning rate scheduling based on validation
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.lr_patience:
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr / args.lr_factor, args.lr_min)
                    param_group['lr'] = new_lr
                    log.info(f'Reducing learning rate from {old_lr} to {new_lr}')
                patience_counter = 0


def get_layer_input_for_adaptation(model, layer_name, data_loader, device):
    """
    Get the cross-input matrix (X^T @ X) for a specific layer
    by capturing activations from the previous layer
    """
    # Map layer names to their input sources
    layer_input_map = {
        'conv1': None,  # Uses raw input
        'conv2': 'conv1',
        'conv3': 'conv2',
        'fc1': 'conv3',
        'fc2': 'fc1',
    }
    
    input_source = layer_input_map.get(layer_name)
    
    # Get the actual layer object
    layer = dict(model.named_modules())[layer_name]
    
    if input_source is None:
        # For conv1, use raw input - already handled by get_inputs
        cross_inputs = get_inputs(model, layer, data_loader, 
                                 max_data_count=float('inf'), device=device)
    else:
        # For other layers, we need to capture the output of the previous layer
        # The get_inputs function already does this via forward hooks
        cross_inputs = get_inputs(model, layer, data_loader,
                                 max_data_count=float('inf'), device=device)
    
    return cross_inputs


def adapt_and_train(model, task_data, previous_task_data, task_id, args, device, log):
    """Adapt layers and train for subsequent tasks"""
    
    log.info(f"Adapting model for task {task_id}...")
    
    # Concatenate all previous task data
    all_prev_x = torch.cat(previous_task_data['train']['x'], dim=0)
    all_prev_y = torch.cat(previous_task_data['train']['y'], dim=0)
    
    log.info(f"Previous data size: {all_prev_x.shape}")
    
    prev_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(all_prev_x, all_prev_y),
        batch_size=args.batch_size_train, shuffle=False)
    
    # Define layers to adapt in order
    layers_to_adapt = [
        ('conv1', model.conv1),
        ('conv2', model.conv2),
        ('conv3', model.conv3),
        ('fc1', model.fc1),
        ('fc2', model.fc2),
    ]
    
    # Adapt each layer sequentially
    for idx, (layer_name, layer) in enumerate(layers_to_adapt):
        log.info(f"Adapting layer {idx+1}/5: {layer_name}")
        
        # Get current layer object (might have been replaced)
        current_layer = dict(model.named_modules())[layer_name]
        
        # Skip if already adapted (has weight_U attribute)
        if hasattr(current_layer, 'weight_U'):
            log.info(f"  Layer {layer_name} already adapted, skipping...")
            continue
        
        # Compute cross-inputs matrix for this layer
        cross_inputs = get_layer_input_for_adaptation(
            model, layer_name, prev_loader, device)
        
        log.info(f"  Cross-inputs shape: {cross_inputs.shape}")
        
        # Create adapted layer
        if isinstance(current_layer, nn.Linear):
            adapted_layer = LinearAdapt(current_layer, cross_inputs, eps_rel_tol=args.eps_1)
            log.info(f"  Created LinearAdapt with rank: {adapted_layer.rank}")
        elif isinstance(current_layer, nn.Conv2d):
            adapted_layer = Conv2dAdapt(current_layer, cross_inputs, eps_rel_tol=args.eps_1)
            log.info(f"  Created Conv2dAdapt with rank: {adapted_layer.rank}")
        else:
            log.info(f"  Skipping unsupported layer type: {type(current_layer)}")
            continue
        
        # Replace layer in model
        replace_layer_by_name(model, layer_name, adapted_layer)
        log.info(f"  Replaced {layer_name} in model")
    
    # Now train only U matrices on current task data
    train_adapted_model(model, task_data, task_id, args, device, log)


def train_adapted_model(model, task_data, task_id, args, device, log):
    """Train only the U parameters of adapted layers"""
    
    log.info(f"Training adapted model on task {task_id}...")
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(task_data['train']['x'], 
                                       task_data['train']['y']),
        batch_size=args.batch_size_train, shuffle=True)
    
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(task_data['valid']['x'], 
                                       task_data['valid']['y']),
        batch_size=args.batch_size_test, shuffle=False)
    
    # Collect only trainable parameters (U matrices and task-specific heads)
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            log.info(f"  Trainable: {name}, shape: {param.shape}")
    
    log.info(f"Total trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()
    
    best_valid_acc = 0
    patience_counter = 0
    
    for epoch in range(args.n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, task_id)
        
        # Validation
        model.eval()
        valid_correct = 0
        valid_total = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                pred = outputs[task_id].argmax(dim=1)
                valid_correct += (pred == target).sum().item()
                valid_total += target.size(0)
        
        valid_acc = 100. * valid_correct / valid_total
        
        log.info(f'Epoch {epoch+1}/{args.n_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Valid Acc: {valid_acc:.2f}%')
        
        # Learning rate scheduling
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.lr_patience:
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr / args.lr_factor, args.lr_min)
                    param_group['lr'] = new_lr
                    log.info(f'Reducing learning rate from {old_lr} to {new_lr}')
                patience_counter = 0


def evaluate_all_tasks(model, data, current_task, device, log, accuracy_matrix):
    """Evaluate on all tasks seen so far and update accuracy matrix"""
    model.eval()
    accs = []
    
    num_tasks = accuracy_matrix.shape[0]  # Get number of tasks from matrix shape
    
    with torch.no_grad():
        for t in range(num_tasks):  # Changed from len(data) to num_tasks
            if t > current_task:
                # Haven't learned this task yet
                accs.append(0.0)
            else:
                test_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(data[t]['test']['x'], 
                                                   data[t]['test']['y']),
                    batch_size=100, shuffle=False)
                
                correct = 0
                total = 0
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    pred = outputs[t].argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
                
                task_acc = 100. * correct / total
                accs.append(task_acc)
        
        # Store accuracies in matrix
        accuracy_matrix[current_task] = accs
        
        # Log current row
        log.info(f"After Task {current_task}:")
        acc_str = "  ".join([f"{acc:5.1f}%" if acc > 0 else "  0.0%" for acc in accs])
        log.info(f"  {acc_str}")
    
    # Calculate average accuracy (only on learned tasks)
    avg_acc = np.mean([acc for acc in accs if acc > 0])
    
    return avg_acc, accs


def compute_metrics(accuracy_matrix, num_tasks):
    """
    Compute final average accuracy and backward transfer
    
    Args:
        accuracy_matrix: numpy array of shape (num_tasks, num_tasks)
                        where accuracy_matrix[i][j] is the accuracy on task j after training on task i
        num_tasks: total number of tasks
    
    Returns:
        avg_acc: average accuracy on all tasks after training on all tasks
        bwt: backward transfer metric
    """
    # Final average accuracy (last row, average of all tasks)
    avg_acc = np.mean(accuracy_matrix[-1])
    
    # Backward Transfer (BWT)
    # BWT = (1/T-1) * sum_{i=1}^{T-1} (acc_{T,i} - acc_{i,i})
    # This measures how much learning new tasks affects performance on previous tasks
    bwt = 0.0
    if num_tasks > 1:
        bwt_sum = 0.0
        for i in range(num_tasks - 1):
            # acc_{T,i}: accuracy on task i after learning all tasks (last row)
            # acc_{i,i}: accuracy on task i right after learning task i (diagonal)
            bwt_sum += (accuracy_matrix[-1][i] - accuracy_matrix[i][i])
        bwt = bwt_sum / (num_tasks - 1)
    
    return avg_acc, bwt


def print_final_results(accuracy_matrix, task_order, log):
    """Print the final accuracy matrix and metrics in the desired format"""
    num_tasks = accuracy_matrix.shape[0]
    
    log.info("\n" + "="*60)
    log.info("Accuracies =")
    
    # Print accuracy matrix
    for i in range(num_tasks):
        acc_str = "  ".join([f"{acc:4.1f}%" if acc > 0 else " 0.0%" for acc in accuracy_matrix[i]])
        log.info(f" {acc_str}")
    
    log.info("-" * 60)
    
    # Print task order
    task_order_str = " ".join([str(t) for t in task_order])
    log.info(f"Task Order : [{task_order_str}]")
    
    # Compute and print metrics
    avg_acc, bwt = compute_metrics(accuracy_matrix, num_tasks)
    log.info(f"Final Avg Accuracy: {avg_acc:.2f}%")
    log.info(f"Backward transfer: {bwt:.2f}%")
    log.info("="*60 + "\n")
    
    return avg_acc, bwt


def main(args):
    """Main training function"""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Import dataloader (adjust this import based on your file structure)
    from dataloader import cifar100 as cf100
    data,taskcla,inputsize=cf100.get(seed=args.seed, pc_valid=args.pc_valid)
    num_tasks = len(taskcla)
    
    # Create logger
    log = create_log_dir(args.savename, f'log_{args.seed}.txt')
    log.info(f"Tasks: {taskcla}")
    log.info(f"Image size: {inputsize}")
    
    # Initialize model
    model = AlexNet(taskcla).to(device)
    log.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Store previous task data for concatenation
    previous_task_data = {'train': {'x': [], 'y': []}}
    
    # Initialize accuracy matrix (num_tasks x num_tasks)
    # accuracy_matrix[i][j] = accuracy on task j after training on task i
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    
    # Task order (for logging)
    task_order = list(range(num_tasks))
    
    # Loop through tasks
    for task_id in range(num_tasks):
        log.info(f"\n{'#'*60}")
        log.info(f"# Starting Task {task_id}: {data[task_id]['name']}")
        log.info(f"# Classes: {data[task_id]['ncla']}")
        log.info(f"# Train samples: {data[task_id]['train']['x'].shape[0]}")
        log.info(f"{'#'*60}\n")
        
        if task_id == 0:
            # First task: train from scratch
            train_first_task(model, data[task_id], args, device, log)
        else:
            # Subsequent tasks: adapt layers then train
            adapt_and_train(model, data[task_id], previous_task_data, 
                          task_id, args, device, log)
        
        # Evaluate on all tasks and update accuracy matrix
        avg_acc, task_accs = evaluate_all_tasks(model, data, task_id, device, log, accuracy_matrix)
        
        # Store current task data for future tasks
        previous_task_data['train']['x'].append(data[task_id]['train']['x'])
        previous_task_data['train']['y'].append(data[task_id]['train']['y'])
        
        log.info(f"\nTask {task_id} completed. Average accuracy so far: {avg_acc:.2f}%\n")
    
    # Print final results in the desired format
    final_avg_acc, final_bwt = print_final_results(accuracy_matrix, task_order, log)
    
    return final_avg_acc, final_bwt

# END DEBUG


# def create_log_dir(path, filename='log.txt'):
#     import logging
#     if not os.path.exists(path):
#         os.makedirs(path)
#     logger = logging.getLogger(path)
#     logger.setLevel(logging.DEBUG)
#     fh = logging.FileHandler(path+'/'+filename)
#     fh.setLevel(logging.DEBUG)
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.DEBUG)
#     logger.addHandler(fh)
#     logger.addHandler(ch)
#     return logger

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    # Set formatter for better readability (optional)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential CIFAR100 with DFGP')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=5, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pc_valid',default=0.05,type=float,
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
    parser.add_argument('--savename', type=str, default='./logs/CIFAR100/',
                        help='save path')
    parser.add_argument('--eps_1', type=float, default=0.01, metavar='Epsilon_1',
                        help='epsilon_1 for SVD')
    parser.add_argument('--mixup_alpha', type=float, default=20, metavar='Alpha',
                        help='mixup_alpha')
    parser.add_argument('--mixup_weight', type=float, default=0.1, metavar='Weight',
                        help='mixup_weight')

    args = parser.parse_args()
    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    # log = create_log_dir(args.savename, 'log_{}.txt'.format(str_time_))
    log = create_log_dir(args.savename, f'log_{str_time_}.txt')

    # for mixup_weight in [0.01, 0.001, 0.0001]:
    #     for thro_ in [0.94, 0.95, 0.96]:

    for mixup_weight in [0.0001]:
        for thro_ in [0.96]:

            accs, bwts = [], []
            args.mixup_weight = mixup_weight
            args.thro = thro_

            str_time = str_time_ + '_' + str(mixup_weight) +  '_' + str(thro_)

            # for seed_ in [1, 2]:
            for seed_ in [1]:
                try:
                    args.seed = seed_
                    log.info('=' * 100)
                    log.info('Arguments =')
                    log.info(str(args))
                    log.info('=' * 100)

                    train_begin_time = time.time()
                    acc, bwt = main(args)
                    print(time.time() - train_begin_time)
                    log.info(f"time cost = {str(time.time() - train_begin_time)}")

                    accs.append(acc)
                    bwts.append(bwt)
                except Exception as e:
                    log.error(f"seed {seed_} Error: {type(e).__name__}: {str(e)}")
                    import traceback
                    log.error(traceback.format_exc())