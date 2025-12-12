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
from torch.utils.data import TensorDataset, DataLoader

from flatness_minima import SAM

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


def test(args, model, device, x, y, criterion, task_id):
    model.eval()
    total_loss = 0
    total_num = 0 
    correct = 0
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # r = torch.LongTensor(r) # DEBUG
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

# Adaptations
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

        if zero_rank == 0:

            self.weight_U = None
            self.weight_V = None
        else:
            V_Xt = V_Xt[:zero_rank, :]
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
    
    @torch.no_grad()
    def merge_weights(self):
        if self.weight_V is None:
            return
        self.old_linear.weight += self.delta_W
        self.reset_parameters()

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

        eps_tol = eps_rel_tol * S_X.sum()
        print(f"{S_X.sum() = }")
        print(f"{eps_rel_tol = } - {eps_tol = }")
        zero_rank = torch.sum(S_X <= eps_tol).item()

        if zero_rank == 0:
            self.weight_U = None
            self.weight_V = None
        else:
            V_Xt = V_Xt[:zero_rank, :]
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

    @torch.no_grad()
    def merge_weights(self):
        if self.weight_V is None:
            return
        delta_W_reshaped = self.delta_W.view_as(self.old_conv.weight)
        self.old_conv.weight += delta_W_reshaped
        self.reset_parameters()

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

# train, train_projected using SGD, Adam, SGD with momentum.

def train(args, model, device, x, y, optimizer, criterion, task_id):
    model.train()
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r)
    
    # Loop batches
    for i in range(0, len(r), args.batch_size_train):
        if i + args.batch_size_train <= len(r): 
            b = r[i:i + args.batch_size_train]
        else: 
            b = r[i:]
        
        data = x[b].to(device)
        target = y[b].to(device)

        optimizer.zero_grad()
        output = model(data)[task_id]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def train_task_i(args, model, device, x, y, optimizer, criterion, task_id):
    model.train()
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r)
    
    # Loop batches
    for i in range(0, len(r), args.batch_size_train):
        if i + args.batch_size_train <= len(r): 
            b = r[i:i+args.batch_size_train]
        else: 
            b = r[i:]
        
        data = x[b].to(device)
        target = y[b].to(device)
        
        optimizer.zero_grad()
        output = model(data)[task_id]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()



def main(args):
    tstart=time.time()
    ## Device Setting 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def get_last_layer_names(model):
        last_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList):
                parent_name = name
                for idx in range(len(module)):
                    last_layers.append(f"{parent_name}.{idx}")
        return last_layers

    ## Load CIFAR100 DATASET
    from dataloader import cifar100 as cf100
    data,taskcla,inputsize=cf100.get(seed=args.seed, pc_valid=args.pc_valid)

    acc_matrix=np.zeros((10,10))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []

    all_previous_x = []  # will store [X_1, X_2, ..., X_{t-1}]
    all_previous_y = []
    for k,ncla in taskcla:
        # threshold = np.array([args.gpm_thro] * 5)

        log.info('*'*100)
        log.info(f'Task {k:2d} ({data[k]["name"]:s})')
        log.info('*'*100)
        xtrain=data[k]['train']['x'].to(device)
        ytrain=data[k]['train']['y'].to(device)
        xvalid=data[k]['valid']['x'].to(device)
        yvalid=data[k]['valid']['y'].to(device)
        xtest =data[k]['test']['x'].to(device)
        ytest =data[k]['test']['y'].to(device)
        task_list.append(k)

        lr = args.lr 
        best_loss=np.inf
        log.info ('-'*40)
        log.info (f'Task ID :{task_id} | Learning Rate : {lr}')
        log.info ('-'*40)
        
        if task_id==0:
            model = AlexNet(taskcla).to(device)
            log.info ('Model parameters ---')
            for k_t, (m, param) in enumerate(model.named_parameters()):
                log.info(f"{k_t}, {m}, {param.shape}")
            log.info ('-'*40)

            best_model=get_model(model)
            if args.optimizer == 'sgd':
                base_optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=args.weight_decay)
            elif args.optimizer == 'sgdm':
                base_optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)
            elif args.optimizer == 'adam':
                base_optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

            optimizer = base_optimizer

            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0=time.time()
                train(args, model, device, xtrain, ytrain, optimizer, criterion, k)
                clock1=time.time()
                tr_loss,tr_acc = test(args, model, device, xtrain, ytrain,  criterion, k)
                log.info(f'Epoch {epoch:3d} | Train: loss={tr_loss:.3f}, acc={tr_acc:5.1f}% | time={1000*(clock1-clock0):5.1f}ms |')
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion, k)
                log.info(f' Valid: loss={valid_loss:.3f}, acc={valid_acc:5.1f}% |')
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=get_model(model)
                    patience=args.lr_patience
                else:
                    patience-=1
                    if patience<=0:
                        lr/=args.lr_factor
                        log.info(f'lr={lr:.1e}')
                        if lr<args.lr_min:
                            break
                        patience=args.lr_patience
                        adjust_learning_rate(optimizer, epoch, args)
                log.info('')
            set_model_(model,best_model)
            # Test
            log.info ('-'*40)
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, k)
            all_previous_x.append(xtrain.clone())
            all_previous_y.append(ytrain.clone())
            log.info(f'Test: loss={test_loss:.3f} , acc={test_acc:5.1f}%')

            # Check weight after training
            log.info('='*60)
            log.info(f'AFTER TASK {task_id} TRAINING - Weight Statistics (with adaptations):')
            for name, module in model.named_modules():
                if 'old_conv' in name or 'old_linear' in name:
                    continue
                if isinstance(module, LinearAdapt):
                    effective_weight = module.old_linear.weight + module.delta_W
                    log.info(f'{name} (LinearAdapt):')
                    log.info(f'old_weight: mean={module.old_linear.weight.data.mean().item():.6f}, norm={module.old_linear.weight.data.norm().item():.6f}')
                    log.info(f'delta_W: mean={module.delta_W.data.mean().item():.6f}, norm={module.delta_W.data.norm().item():.6f}')
                    log.info(f'effective (old+delta): mean={effective_weight.data.mean().item():.6f}, norm={effective_weight.data.norm().item():.6f}')
                elif isinstance(module, Conv2dAdapt):
                    effective_weight = module.old_conv.weight + module.delta_W.view_as(module.old_conv.weight)
                    log.info(f'{name} (Conv2dAdapt):')
                    log.info(f'old_weight: mean={module.old_conv.weight.data.mean().item():.6f}, norm={module.old_conv.weight.data.norm().item():.6f}')
                    log.info(f'delta_W: mean={module.delta_W.data.mean().item():.6f}, norm={module.delta_W.data.norm().item():.6f}')
                    log.info(f'effective (old+delta): mean={effective_weight.data.mean().item():.6f}, norm={effective_weight.data.norm().item():.6f}')
                elif isinstance(module, (nn.Linear, nn.Conv2d)) and not name.startswith('fc3'):
                    log.info(f'{name} ({type(module).__name__}):')
                    log.info(f'weight: mean={module.weight.data.mean().item():.6f}, norm={module.weight.data.norm().item():.6f}')
            log.info('='*60)

        # elif task_id < args.debug_task_id:

        else:  # task_id > 0
            log.info('Adapting model for new task...')

            last_layer_names = get_last_layer_names(model)
            log.info(f'Detected last layer names: {last_layer_names}')
            
            # Check weight before training
            log.info('='*60)
            log.info(f'BEFORE TASK {task_id} (before consolidation) - Weight Statistics:')
            for name, module in model.named_modules():
                # Skip last layers in this check
                is_last_layer = any(name.startswith(ll) for ll in last_layer_names)
                if is_last_layer:
                    continue
                    
                if isinstance(module, LinearAdapt):
                    effective_weight = module.old_linear.weight + module.delta_W
                    log.info(f'{name} (LinearAdapt - from previous task):')
                    log.info(f'  effective weight: mean={effective_weight.data.mean().item():.6f}, norm={effective_weight.data.norm().item():.6f}')
                elif isinstance(module, Conv2dAdapt):
                    effective_weight = module.old_conv.weight + module.delta_W.view_as(module.old_conv.weight)
                    log.info(f'{name} (Conv2dAdapt - from previous task):')
                    log.info(f'  effective weight: mean={effective_weight.data.mean().item():.6f}, norm={effective_weight.data.norm().item():.6f}')
            log.info('='*60)
            
            X_prev = torch.cat(all_previous_x, dim=0)
            Y_prev = torch.cat(all_previous_y, dim=0)
            log.info(f'Previous data shape: {X_prev.shape}, {Y_prev.shape}')
            
            layers_to_adapt = []
            for name, module in model.named_modules():
                if 'old_conv' in name or 'old_linear' in name:
                    continue
                
                is_last_layer = any(name.startswith(ll) for ll in last_layer_names)
                
                if isinstance(module, (nn.Linear, nn.Conv2d)) and not is_last_layer:
                    layers_to_adapt.append(name)
                elif isinstance(module, (LinearAdapt, Conv2dAdapt)) and not is_last_layer:
                    layers_to_adapt.append(name)

            log.info(f'Layers to adapt: {layers_to_adapt}')
            
            # For each layer, compute U and V, then replace with adapted version
            for layer_idx, layer_name in enumerate(layers_to_adapt):
                log.info(f'Processing layer {layer_idx + 1}: {layer_name}')
                
                layer_dict = dict(model.named_modules())
                
                if layer_name not in layer_dict:
                    log.info(f'Warning: Layer {layer_name} not found in model')
                    continue
                else:
                    layer = layer_dict[layer_name]
                
                if isinstance(layer, (LinearAdapt, Conv2dAdapt)):
                    log.info(f'Layer {layer_name} is already adapted, merging weights')
                    layer.merge_weights()
                    
                    if isinstance(layer, LinearAdapt):
                        new_base_layer = layer.old_linear
                    else:
                        new_base_layer = layer.old_conv
                    
                    replace_layer_by_name(model, layer_name, new_base_layer)
                    
                    layer = new_base_layer
                    log.info(f'Consolidated weights into new base layer for {layer_name}')
                
                layer_dict = dict(model.named_modules())
                layer = layer_dict[layer_name]
                
                prev_dataset = TensorDataset(X_prev, Y_prev)
                prev_loader = DataLoader(prev_dataset, batch_size=args.batch_size_train, shuffle=False)
                
                log.info(f'Computing cross inputs for {layer_name}...')
                cross_inputs = get_inputs(
                    network=model,
                    layer=layer,
                    data_loader=prev_loader,
                    max_data_count=len(X_prev),
                    device=device
                )
                
                if cross_inputs is None:
                    log.info(f'Warning: Failed to compute cross inputs for {layer_name}')
                    continue
                    
                log.info(f'Cross inputs shape: {cross_inputs.shape}')
                
                # Replace layer with adapted version
                if isinstance(layer, nn.Linear):
                    adapted_layer = LinearAdapt(layer, cross_inputs, eps_rel_tol=args.eps_1)
                    log.info(f'Created LinearAdapt with rank: {adapted_layer.rank}')
                elif isinstance(layer, nn.Conv2d):
                    adapted_layer = Conv2dAdapt(layer, cross_inputs, eps_rel_tol=args.eps_1)
                    log.info(f'Created Conv2dAdapt with rank: {adapted_layer.rank}')
                else:
                    log.info(f'Skipping layer {layer_name} (not Linear or Conv2d)')
                    continue

                replace_layer_by_name(model, layer_name, adapted_layer)
                log.info(f'Replaced layer {layer_name} with adapted version')
            
            log.info('-' * 40)
            log.info('Setting up optimizer for U parameters and current task head...')

            # Check all existed parameters
            log.info('All model parameters:')
            for name, param in model.named_parameters():
                log.info(f'  {name}, shape: {param.shape}, requires_grad: {param.requires_grad}')

            for name, param in model.named_parameters():
                param.requires_grad = False

            trainable_params = []
            # Extract the base name of last layer (e.g "fc3" from "fc3.0")
            last_layer_base = last_layer_names[0].rsplit('.', 1)[0]
            current_task_head_name = f"{last_layer_base}.{k}"  # e.g., "fc3.1"

            for name, param in model.named_parameters():
                # Enable weight_U parameters (but not from last layers)
                is_from_last_layer = any(name.startswith(ll) for ll in last_layer_names)
                if 'weight_U' in name and not is_from_last_layer:
                    param.requires_grad = True
                    trainable_params.append(param)
                    log.info(f'Trainable parameter (U): {name}, shape: {param.shape}')
                # Check if parameter belongs to current task head (e.g., "fc3.1.weight")
                elif name.startswith(f"{current_task_head_name}."):
                    param.requires_grad = True
                    trainable_params.append(param)
                    log.info(f'Trainable parameter (task head - FULL): {name}, shape: {param.shape}')
            
            if len(trainable_params) == 0:
                log.info('Warning: No trainable parameters found!')
                if args.optimizer == 'sgd':
                    base_optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                elif args.optimizer == 'sgdm':
                    base_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
                elif args.optimizer == 'adam':
                    base_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            else:
                log.info(f'Number of trainable parameters: {len(trainable_params)}')
                log.info(f'Total trainable params count: {sum(p.numel() for p in trainable_params)}')
                if args.optimizer == 'sgd':
                    base_optimizer = optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
                elif args.optimizer == 'sgdm':
                    base_optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
                elif args.optimizer == 'adam':
                    base_optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
            
            optimizer = base_optimizer
            
            log.info('-' * 40)
            
            # Training loop
            best_loss = np.inf
            best_model = get_model(model)
            patience = args.lr_patience
            lr = args.lr
            
            for epoch in range(1, args.n_epochs + 1):
                # Train
                clock0 = time.time()
                train_task_i(args, model, device, xtrain, ytrain, optimizer, criterion, k)
                clock1 = time.time()
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain, criterion, k)
                log.info(f'Epoch {epoch:3d} | Train: loss={tr_loss:.3f}, acc={tr_acc:5.1f}% | time={1000*(clock1-clock0):5.1f}ms |')
                
                # Validate
                valid_loss, valid_acc = test(args, model, device, xvalid, yvalid, criterion, k)
                log.info(f' Valid: loss={valid_loss:.3f}, acc={valid_acc:5.1f}% |')
                
                # Adapt lr
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = get_model(model)
                    patience = args.lr_patience
                else:
                    patience -= 1
                    if patience <= 0:
                        lr /= args.lr_factor
                        log.info(f'lr={lr:.1e}')
                        if lr < args.lr_min:
                            break
                        patience = args.lr_patience
                        adjust_learning_rate(optimizer, epoch, args)
                log.info('')
            
            set_model_(model, best_model)
            
            # Test
            log.info('-' * 40)
            test_loss, test_acc = test(args, model, device, xtest, ytest, criterion, k)
            log.info(f'Test: loss={test_loss:.3f} , acc={test_acc:5.1f}%')

            # Check weight after training
            log.info('='*60)
            log.info(f'AFTER TASK {task_id} TRAINING - Weight Statistics (with adaptations):')
            for name, module in model.named_modules():
                if 'old_conv' in name or 'old_linear' in name:
                    continue
                if isinstance(module, LinearAdapt):
                    effective_weight = module.old_linear.weight + module.delta_W
                    log.info(f'{name} (LinearAdapt):')
                    log.info(f'  old_weight: mean={module.old_linear.weight.data.mean().item():.6f}, norm={module.old_linear.weight.data.norm().item():.6f}')
                    log.info(f'  delta_W: mean={module.delta_W.data.mean().item():.6f}, norm={module.delta_W.data.norm().item():.6f}')
                    log.info(f'  effective (old+delta): mean={effective_weight.data.mean().item():.6f}, norm={effective_weight.data.norm().item():.6f}')
                elif isinstance(module, Conv2dAdapt):
                    effective_weight = module.old_conv.weight + module.delta_W.view_as(module.old_conv.weight)
                    log.info(f'{name} (Conv2dAdapt):')
                    log.info(f'  old_weight: mean={module.old_conv.weight.data.mean().item():.6f}, norm={module.old_conv.weight.data.norm().item():.6f}')
                    log.info(f'  delta_W: mean={module.delta_W.data.mean().item():.6f}, norm={module.delta_W.data.norm().item():.6f}')
                    log.info(f'  effective (old+delta): mean={effective_weight.data.mean().item():.6f}, norm={effective_weight.data.norm().item():.6f}')
                elif isinstance(module, (nn.Linear, nn.Conv2d)):
                    log.info(f'{name} ({type(module).__name__}):')
                    log.info(f'  weight: mean={module.weight.data.mean().item():.6f}, norm={module.weight.data.norm().item():.6f}')
            log.info('='*60)
            
            all_previous_x.append(xtrain.clone())
            all_previous_y.append(ytrain.clone())

        # else:
        #     break  # Uncomment if you used else task_id < debug_task_id
        
        # save accuracy
        jj = 0 
        for ii in np.array(task_list)[0:task_id+1]:
            xtest =data[ii]['test']['x'].to(device)
            ytest =data[ii]['test']['y'].to(device)
            _, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion,ii) 
            jj +=1
        log.info('Accuracies =')
        for i_a in range(task_id + 1):
            acc_ = ''
            for j_a in range(acc_matrix.shape[1]):
                acc_ += f'{acc_matrix[i_a, j_a]:5.1f}% '
            log.info(acc_)
        # update task id 
        task_id +=1
    log.info('-'*50)
    # Simulation Results 
    log.info (f'Task Order : {np.array(task_list)}')
    log.info (f'Final Avg Accuracy (for seed {args.seed}): {acc_matrix[-1].mean():5.2f}%')
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    log.info (f'Backward transfer: {bwt:5.2f}%')
    log.info(f'[Elapsed time = {(time.time()-tstart)*1000:.1f} ms]')
    log.info('-'*50)
    return acc_matrix[-1].mean(), bwt


def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    # fh = logging.FileHandler(path+'/'+filename)
    # fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential CIFAR100 with NESS')
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
    parser.add_argument('--gpm_thro', type=float, default=0.95, metavar='gradient projection',
                        help='gpm_thro')
    # NESS parameters
    parser.add_argument('--eps_1', type=float, default=0.01, metavar='Epsilon_1',
                        help='epsilon_1 for SVD')
    parser.add_argument('--debug_task_id',default=3,type=float,
                        help='fraction of training data used for validation')
    parser.add_argument('--optimizer', type=str, default='sgdm',
                        help='optimizer type: sgdm, adam, sgd')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay for optimizer')
    
    args = parser.parse_args()
    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log = create_log_dir(args.savename, f'log_our_{str_time_}.txt')


    accs, bwts = [], []

    for seed_ in [1, 2, 3, 4, 37]:
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