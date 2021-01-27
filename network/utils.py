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
