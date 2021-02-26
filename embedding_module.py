import torch
import torch.nn.functional as F
import numpy as np
from network.embedding import EmbeddingNet

class EmbeddingModule(object):

    def __init__(self, data_sizes, emb_dim, filters, kernels, strides, paddings, dilations, fc_layers, device, margin=0.1):

        self.net = EmbeddingNet(list(data_sizes['img_shape'][:2]) + [data_sizes['img_shape'][2] * data_sizes['frames']],
                       emb_dim, filters, kernels, strides, paddings, dilations, fc_layers,
                       norm_conv='layer', norm_fc=None, act='elu', drop_rate_conv=0.0, drop_rate_fc=0.0)
        self.net.to(device)
        # self.net = torch.nn.DataParallel(self.net, device_ids=[0])
        print("emb_net params")
        for name, param in self.net.named_parameters():
            print(name, param.shape)
        
        self.device = device
        self.margin = margin
        self.emb_dim = emb_dim
        self.support_size = data_sizes['support_size']
        self.support_query_size = 0 # overriden in forward() 

    def parameters(self):
        return self.net.parameters()

    def eval(self):
        self.net.eval()

    def train(self, train=True):
        self.net.train(train)

    def save(self, model_path):
        torch.save(self.net.state_dict(), model_path)

    def load(self, model_path, device):
        self.net.load_state_dict(torch.load(model_path, map_location=device))

    def _cos_hinge_loss(self, U_s, q_s, margin=0.1):
        """
        :param U_s: mean support embeddings of shape (N, emb_dim)
        :param q_s: query embeddings of shape (N, q_n, emb_dim)

        :return loss: scalar
        :return accuracy: scalar
        """
        N = q_s.shape[0]
        q_s = q_s.reshape(-1, q_s.shape[-1]) # (N * q_n, emb_dim)
        
        # Similarities of every support mean sentence with every query sentence
        similarities = torch.mm(U_s, torch.transpose(q_s, 0, 1)) # (N, N * q_n)
        similarities = similarities.view(N, N, -1) # (N, N, q_n)

        # Gets the diagonal
        mask_pos = torch.eye(N) == 1
        positives = similarities[mask_pos, :] # (N, q_n)
        positives_ex = positives.view(N, 1, -1) # (N, 1, q_n)

        # Gets everything but the diagonal
        mask_neg = torch.eye(N) == 0
        negatives = similarities[mask_neg, :] # (N * (N - 1), q_n)
        negatives = negatives.view(N, N - 1, -1) # (N, N - 1, q_n)

        # loss
        loss = torch.clamp(margin - positives_ex + negatives, min=0.0)
        loss = torch.mean(loss)

        # accuracy
        max_of_negs = torch.max(negatives, dim=1).values  # (N, q_n)
        accuracy = torch.greater(positives, max_of_negs).float()
        accuracy = torch.mean(accuracy)

        return loss, accuracy

    def _norm(self, vecs, dim=1):
        mag = torch.sqrt(torch.sum(torch.square(vecs), dim, keepdim=True))
        return vecs / torch.clamp(mag, min=1e-6)

    def _compute_embeddings(self, embnet_images):
        """
        :param embnet_images: shape (N, U_n + q_n, frames, h, w, c)

        :return sentences: shape (N, U_n + q_n, emb_dim)
        """
        embnet_images = torch.cat(torch.unbind(embnet_images, dim=2), dim=-1) # (N, U_n + q_n, h, w, c * frames)
        embnet_images = embnet_images.view([-1] + list(embnet_images.shape[2:])) # (N * (U_n + q_n), h, w, c * frames)
        
        sentences = self.net(embnet_images) # (N * (U_n + q_n), emb_dim)
        sentences = sentences.view(-1, self.support_query_size, self.emb_dim) # (N, U_n + q_n, emb_dim)

        return sentences

    def forward(self, inputs, eval=False):
        # load data
        embnet_images = inputs['embnet_images'].to(self.device) # (N, U_n + q_n, frames, img_shape) where img_shape = (h, w, c)
        self.support_query_size = embnet_images.shape[1]

        # compute embeddings
        sentences = self._compute_embeddings(embnet_images) # (N, U_n + q_n, emb_dim)
        
        if eval:
            return {
                'sentences': sentences,
                'support_embeddings': self._norm(torch.mean(self._norm(sentences, dim=2), dim=1), dim=1) # (N, emb_dim)
            }

        # embedding loss
        q_s = sentences[:, self.support_size:] # (N, q_n, emb_dim)
        U_s = sentences[:, :self.support_size] # (N, U_n, emb_dim)
        U_s = self._norm(torch.mean(self._norm(U_s, dim=2), dim=1), dim=1) # (N, emb_dim)
        loss_emb, acc_emb = self._cos_hinge_loss(U_s, q_s, margin=self.margin)

        return {
            'support_embeddings': U_s,
            'loss_emb': loss_emb,
            'acc_emb': acc_emb,
        }