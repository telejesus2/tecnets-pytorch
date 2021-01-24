import torch
import torch.nn.functional as F
import numpy as np

class MetaLearner(object):

    def __init__(self, emb_net, ctr_net, train_gen, val_gen, optimizer, device):
        self.emb_net = emb_net
        self.ctr_net = ctr_net
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.optimizer = optimizer
        self.device = device

    def mse_loss(actions, labels):
        """
        :param actions: shape (N, examples, action_dim)
        :param labels: shape (N, examples, action_dim)
        """
        support_size = actions.shape[1] // 2
        loss_support = F.mse_loss(actions[:, :support_size], labels[:, :support_size], reduction='mean')
        loss_query = F.mse_loss(actions[:, support_size:], labels[:, support_size:], reduction='mean')

        loss_support = self.support_lambda * loss_support
        loss_query = self.query_lambda * loss_query
        return loss_support, loss_query

    def margin_loss(U_s, q_s, margin=0.1):
        """
        :param U_s: mean support embeddings of shape (N, emb_dim)
        :param q_s: query embeddings of shape (N, q_n, emb_dim)
        """
        N = q_s.shape[0]
        q_s = q_s.view(-1, q_s.shape[-1]) # (N * q_n, emb_dim)
        
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

        loss = torch.clamp(margin - positives_ex + negatives, min=0.0)
        loss = torch.mean(loss)

        # loss
        self.loss_embedding = self.loss_lambda * loss

        # accuracy
        max_of_negs = torch.max(negatives, dim=1).values  # (N, q_n)
        accuracy = torch.greater(positives, max_of_negs).float()
        accuracy = torch.mean(accuracy)

        return loss, accuracy

    def _norm(vecs, dim=1):
        mag = torch.sqrt(torch.sum(torch.square(vecs), dim, keepdim=True))
        return vecs / torch.clamp(mag, min=1e-6)

    def meta_train(self, epoch):
        loss_emb_list, loss_ctr_U_list, loss_ctr_q_list, loss_list = [], [], [], []

        for i, tasks in enumerate(self.train_gen):

            loss_emb, loss_ctr_U, loss_ctr_q = 0, 0, 0

            # compute embeddings
            embnet_images = tasks['embnet_images'] # (N, U_n + q_n, frames, img_shape) where img_shape = (h, w, c)
            batch_size, support_query_size = embnet_images.shape[:2]
            embnet_images = torch.cat(torch.unbind(embnet_images, dim=2), dim=-1) # (N, U_n + q_n, h, w, c * frames)
            embnet_images = embnet_images.view([-1] + list(embnet_images.shape[2:])) # (N * (U_n + q_n), h, w, c * frames)
            
            sentences = emb_net(embnet_images) # (N * (U_n + q_n), emb_dim)
            sentences = sentences.view(batch_size, support_query_size, -1) # (N, U_n + q_n, emb_dim)

            q_s = sentences[:, support_size:] # (N, q_n, emb_dim)
            U_s = sentences[:, :support_size] # (N, U_n, emb_dim)
            U_s = _norm(torch.mean(_norm(U_s, dim=2), dim=1), dim=1) # (N, emb_dim)
            emb_dim = U_s.shape[-1]

            # embedding loss
            loss, accuracy = margin_loss(U_s, q_s)
     
            # compute actions
            ctrnet_images = tasks['ctrnet_images'] # (N, examples, img_shape) where img_shape = (h, w, c)
            examples = ctrnet_images.shape[1]
            ctrnet_images = ctrnet_images.view([-1] + list(ctrnet_images.shape[2:])) # (N * examples, h, w, c)

            U_s = U_s.view(-1, 1, emb_dim).expand(-1, examples, emb_dim) # (N, examples, emb_dim)
            U_s = U_s.view(-1, emb_dim) # (N * examples, emb_dim)
    
            states = tasks['ctrnet_states'] # (N, examples, state_dim)
            states = states.view(-1, states.shape[-1]) # (N * examples, state_dim)

            actions_pred = ctr_net(ctrnet_images, U_s, states) # (N * examples, action_dim)
    
            actions_pred = actions_pred.view(-1, examples, actions_pred[-1]) # (N, examples, action_dim)

            # control loss
            actions_labels = tasks['ctrnet_actions'] # (N, examples, action_dim)
            loss_ctr_U, loss_ctr_q = mse_loss(actions_pred, actions_labels)
            
