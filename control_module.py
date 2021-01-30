import torch
import torch.nn.functional as F
import numpy as np
from network.control import ControlNet

class ControlModule(object):

    def __init__(self, data_sizes, emb_dim, filters, kernels, strides, paddings, dilations, fc_layers, device):

        self.ctr_net = ControlNet(data_sizes['img_shape'], data_sizes['state_dim'], data_sizes['action_dim'],
                       emb_dim, filters, kernels, strides, paddings, dilations, fc_layers,
                       norm_conv='layer', norm_fc=None, act='elu', drop_rate_conv=0.0, drop_rate_fc=0.0)
        self.ctr_net.to(device)
        # self.ctr_net = torch.nn.DataParallel(self.ctr_net, device_ids=[0])
        print("ctr_net params")
        for name, param in self.ctr_net.named_parameters():
            print(name, param.shape)

        self.device = device
        self.emb_dim = emb_dim
        self.examples_size = 0 # overriden in forward()

    def parameters(self):
        return self.ctr_net.parameters()

    def eval(self):
        self.ctr_net.eval()

    def train(self, train=True):
        self.ctr_net.train(train)

    def save(self, model_path):
        torch.save(self.ctr_net.state_dict(), model_path)

    def load(self, model_path, device):
        self.ctr_net.load_state_dict(torch.load(model_path, map_location=device))

    def _mse_loss(self, actions, labels):
        """
        :param actions: shape (N, k, action_dim)
        :param labels: shape (N, k, action_dim)

        :return: scalar
        """
        return F.mse_loss(actions, labels, reduction='mean')

    def _compute_actions(self, U_s, ctrnet_images, states):
        """
        :param U_s: shape (N, emb_dim)
        :param ctrnet_images: shape (N, examples_size, h, w, c)
        :param states: shape (N, examples_size, state_dim)

        :return actions: shape (N, examples_size, action_dim)
        """
        ctrnet_images = ctrnet_images.view([-1] + list(ctrnet_images.shape[2:])) # (N * examples_size, h, w, c)

        U_s = U_s.view(-1, 1, self.emb_dim).expand(-1, self.examples_size, self.emb_dim) # (N, examples_size, emb_dim)
        U_s = U_s.reshape(-1, self.emb_dim) # (N * examples_size, emb_dim)

        states = states.view(-1, states.shape[-1]) # (N * examples_size, state_dim)

        actions = self.ctr_net(ctrnet_images, U_s, states) # (N * examples_size, action_dim)
        actions = actions.view(-1, self.examples_size, actions.shape[-1]) # (N, examples_size, action_dim)

        return actions

    def forward(self, inputs, eval=False):
        # load data
        ctrnet_images = inputs['ctrnet_images'].to(self.device) # (N, examples_size, img_shape) where img_shape = (h, w, c)  
        states = inputs['ctrnet_states'].float().to(self.device) # (N, examples_size, state_dim)
        U_s = inputs['support_embeddings'] # (N, emb_dim)
        self.examples_size = ctrnet_images.shape[1]

        # compute actions
        actions_pred = self._compute_actions(U_s, ctrnet_images, states) # (N, examples_size, action_dim)

        if eval:
            return {
                'actions': actions_pred,
            }

        # control loss
        actions_labels = inputs['ctrnet_actions'].float().to(self.device) # (N, examples_size, action_dim)
        loss_ctr_U = self._mse_loss(actions_pred[:, :self.examples_size//2], actions_labels[:, :self.examples_size//2])
        loss_ctr_q = self._mse_loss(actions_pred[:, self.examples_size//2:], actions_labels[:, self.examples_size//2:])

        return {
            'loss_ctr_U': loss_ctr_U,
            'loss_ctr_q': loss_ctr_q,
        }

