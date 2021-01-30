import os
import numpy as np
import random
from data.dataloader import MetaDataLoader
from torch.utils.data._utils.collate import default_collate

class EvalEmbedding(object):

    def __init__(self, dataset, num_subtasks, num_demos):
        self.dataset = dataset
        self.num_subtasks = num_subtasks
        self.num_demos = num_demos

        loader_params = {'batch_size': num_subtasks,
                        'replacement': False, 
                        'shuffle': True}
        self.gen = MetaDataLoader(dataset, num_demos, 0, 0, **loader_params)

    def evaluate(self, epoch, emb_mod, writer=None):
        inputs = next(iter(self.gen))
        embnet_images = inputs['embnet_images'] # (N, U_n + q_n, frames, h, w, c)

        label_img = embnet_images[:,:,1,:] # (N, U_n + q_n, h, w, c)
        label_img = label_img.view([-1] + list(label_img.shape[2:])) # (N * (U_n + q_n), h, w, c)
        label_img = label_img.permute(0,3,1,2) # (N * (U_n + q_n), c, h, w)
        label_img = self.dataset.postprocess_image(label_img)

        sentences = emb_mod.forward(inputs, eval=True)['sentences'] # (N, U_n + q_n, emb_dim)
        sentences = sentences.view(-1, sentences.shape[-1]) # (N * (U_n + q_n), emb_dim)

        meta = []
        for i in range(self.num_subtasks):
            meta += [i] * self.num_demos

        if writer is not None:
            writer.add_embedding(sentences, metadata=meta, label_img=label_img, global_step=epoch, tag='embedding')


