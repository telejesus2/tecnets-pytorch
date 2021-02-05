import numpy as np
import torch.nn as nn

def padding_same(conv_in_shape, kernel_sizes, stride_sizes, dilation_sizes):
    """
        :param conv_in_shape: (h, w)
    """
    if isinstance(kernel_sizes[0], int): kernel_sizes = [(x, x) for x in kernel_sizes]
    if isinstance(stride_sizes[0], int): stride_sizes = [(x, x) for x in stride_sizes]
    if isinstance(dilation_sizes[0], int): dilation_sizes = [(x, x) for x in dilation_sizes]

    h = conv_in_shape[0]
    w = conv_in_shape[1]
    padding_sizes = []
    for (k, s, d) in zip(kernel_sizes, stride_sizes, dilation_sizes):
        pad_h = int(np.ceil((s[0]*(h - 1) - h + d[0]*(k[0] - 1) + 1) / 2))
        pad_w = int(np.ceil((s[1]*(w - 1) - w + d[1]*(k[1] - 1) + 1) / 2))
        padding_sizes.append((pad_h, pad_w))
    return padding_sizes

def conv_shape(conv_in_shape, kernel_sizes, padding_sizes, stride_sizes, dilation_sizes):
    """
        :param conv_in_shape: (h, w)
    """
    if isinstance(kernel_sizes[0], int): kernel_sizes = [(x, x) for x in kernel_sizes]
    if isinstance(padding_sizes[0], int): padding_sizes = [(x, x) for x in padding_sizes]
    if isinstance(stride_sizes[0], int): stride_sizes = [(x, x) for x in stride_sizes]
    if isinstance(dilation_sizes[0], int): dilation_sizes = [(x, x) for x in dilation_sizes]

    h = conv_in_shape[0]
    w = conv_in_shape[1]
    conv_h = [h]
    conv_w = [w]
    for (k, p, s, d) in zip(kernel_sizes, padding_sizes, stride_sizes, dilation_sizes):
        h = int(np.floor((h + 2*p[0] - d[0]*(k[0] - 1) - 1) / s[0] + 1))
        w = int(np.floor((w + 2*p[1] - d[1]*(k[1] - 1) - 1) / s[1] + 1))
        conv_h.append(h)
        conv_w.append(w)
    return (conv_h, conv_w)

def normalization(fn_name, input_shape):
    """
        :param input_shape: (C) or (C,L) or (C,H,W)
    """
    num_channels = input_shape[0]
    fn = None
    if fn_name == 'batch':
        if len(input_shape)==3:
            fn = nn.BatchNorm2d(num_channels) # Input: (N,C,H,W)
        else:
            fn = nn.BatchNorm1d(num_channels) # Input: (N,C) or (N,C,L)            
    elif fn_name == 'instance':
        if len(input_shape)==2:
            fn = nn.InstanceNorm1d(num_channels) # Input: (N,C,L)
        elif len(input_shape)==3:
            fn = nn.InstanceNorm2d(num_channels) # Input: (N,C,H,W) 
    elif fn_name == 'layer':
        from_axis = 1
        fn = nn.LayerNorm(input_shape[from_axis:]) # Input: (N,∗)
    elif fn_name == 'group':
        num_groups = 1 # num_groups = 1: equivalent to LayerNorm along all axes: nn.LayerNorm(input_shape)
                       # num_groups = num_channels: equivalent to either InstanceNorm1d(num_channels) or InstanceNorm2d(num_channels)
        fn = nn.GroupNorm(num_groups, num_channels) # Input: (N,C,*)
    return fn

def activation(fn_name):
    fn = None
    if fn_name == 'relu':
        fn = nn.ReLU()
    elif fn_name == 'elu':
        fn = nn.ELU()
    elif fn_name == 'leaky_relu':
        fn = nn.LeakyReLU()
    return fn

def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return np


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

def plot_grad_flow(named_parameters, fname):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    lgd = plt.legend([Line2D([0], [0], color="c", lw=4),
                      Line2D([0], [0], color="b", lw=4),
                      Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(fname + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

def _iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            print(fn)
            assert all(t is None or torch.all(~torch.isnan(t)) for t in grad_input), f"{fn} grad_input={grad_input} grad_output={grad_output}"
            assert all(t is None or torch.all(~torch.isnan(t)) for t in grad_output), f"{fn} grad_input={grad_input} grad_output={grad_output}"
            
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    _iter_graph(var.grad_fn, hook_cb)