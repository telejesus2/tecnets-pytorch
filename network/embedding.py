"""
Input : 125 × 125 RGB images and the robot joint angles, joint velocities, and end-effector pose
Output : velocities applied to the 6 joints of a Kinova Mico 6-DoF arm.
The proprioceptive data is concatenated to the features extracted from the CNN layers
of the control network, before being sent through the fully-connected layers. 
Input task-embedding : images only
Output task-embedding : vector of length 20.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

from network.utils import conv_shape, normalization, activation

class EmbeddingNet(nn.Module):
    def __init__(self, img_shape,
                 emb_dim, filters, kernels, strides, paddings, dilations, fc_layers,
                 norm_conv='layer', norm_fc='group', act='elu', drop_rate_conv=0.0, drop_rate_fc=0.0):
        """Convolution layers followed by linear layers

        Input to convolution layers: an image (of shape img_shape)
        Input to linear layers: the flattened convolution output
        Output: a vector (of dim emb_dim)

        :param img_shape: (h, w, c)
        """  
        super(EmbeddingNet, self).__init__()
        
        ### activation function
        self.act_fn = activation(act)  

        ### convolution layers   
        num_conv_layers = len(kernels)
        channels = [img_shape[2]] + filters 
        conv_h, conv_w = conv_shape(img_shape[:2], kernels, paddings, strides, dilations)     
        self.conv = nn.ModuleList()
        for i in range(num_conv_layers):
            self.conv.append(nn.Conv2d(
                channels[i], channels[i+1], kernel_size=kernels[i], stride=strides[i], padding=paddings[i]))
        self.ln_conv = nn.ModuleList()
        for i in range(num_conv_layers):
            self.ln_conv.append(normalization(norm_conv, [channels[i+1], conv_h[i+1], conv_w[i+1]]))
        self.drop_conv = nn.Dropout(drop_rate_conv, inplace=True) if drop_rate_conv > 0 else None

        ### linear layers   
        num_fc_layers = len(fc_layers)
        fc_nodes = [channels[-1] * conv_h[-1] * conv_w[-1]] + fc_layers     
        self.fc = nn.ModuleList()
        for i in range(num_fc_layers):
            self.fc.append(nn.Linear(fc_nodes[i], fc_nodes[i+1]))   
        self.ln_fc = nn.ModuleList()
        for i in range(num_fc_layers):
            self.ln_fc.append(normalization(norm_fc, [fc_nodes[i+1]]))
        self.drop_fc = nn.Dropout(drop_rate_fc, inplace=True) if drop_rate_fc > 0 else None

        self.out = nn.Linear(fc_nodes[-1], emb_dim)

        ### weight initialization      
        self._init_weights()

    def forward(self, vision):
        """
        :param vision: shape (N, h, w, c)
        """ 
        x = vision.permute(0,3,1,2) # (N, c, h, w)
        
        for conv, ln in zip(self.conv, self.ln_conv):
            x = self.act_fn(ln(conv(x))) if ln is not None else self.act_fn(conv(x))
            if self.drop_conv is not None: self.drop_conv(x)
            
        x = torch.flatten(x, 1) # (N, channels[-1] * conv_h[-1] * conv_w[-1])

        for fc, ln in zip(self.fc, self.ln_fc):
            x = self.act_fn(ln(fc(x))) if ln is not None else self.act_fn(fc(x))
            if self.drop_fc is not None: self.drop_fc(x)

        x = self.out(x) # (N, emb_dim)

        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                # m.register_parameter('weight', None)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                init.constant_(m.weight, 1)
                # m.register_parameter('weight', None)
                init.constant_(m.bias, 0)