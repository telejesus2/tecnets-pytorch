import numpy as np
import torch.nn as nn

def conv_shape(conv_in_shape, kernel_sizes, padding_sizes, stride_sizes):
    """
        :param conv_in_shape: (h, w)
    """
    h = conv_in_shape[0]
    w = conv_in_shape[1]
    conv_h = [h]
    conv_w = [w]
    for (ksize, pad, stride) in zip(kernel_sizes, padding_sizes, stride_sizes):
        h = int(np.floor((h + 2*pad - (ksize - 1) -1) / stride + 1))
        w = int(np.floor((w + 2*pad - (ksize - 1) -1) / stride + 1))
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
