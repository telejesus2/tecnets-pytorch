import os
import numpy as np
from torch.utils.data._utils.collate import default_collate

class EvalSim(object):

    def __init__(self, dataset, num_subtasks, num_trials, support_size,
                 log_dir=".", record_gifs=False, render=True):
        self.render = render
        self.record_gifs = record_gifs
        self.record_gifs_dir = os.path.join(log_dir, 'evaluated_gifs')

        self.dataset = dataset
        self.demos = dataset.data
        self.num_trials = num_trials
        self.num_subtasks = num_subtasks
        self.time_horizon = dataset.time_horizon
        self.support_size = support_size

    def get_embeddings(self, subtask_index, support_demo_indices, emb_mod):
        inputs = [self.dataset[[subtask_index, demo_indices, []]] for demo_indices in support_demo_indices]
        inputs = default_collate(inputs)
        embedding = emb_mod.forward(inputs, eval=True)['support_embeddings']
        return embedding

    def get_action(self, obs, state, embedding, ctr_mod):
        inputs = {
            'ctrnet_images': np.array([obs]),
            'ctrnet_states': np.array([state]),
            'support_embeddings': embedding,
        }
        inputs = default_collate([inputs])
        action = ctr_mod.forward(inputs, eval=True)['actions']
        return action.cpu().detach().numpy()

    def evaluate(self, epoch, emb_mod, ctr_mod):
        pass

    def save_gifs(self, observations, epoch, subtask_id, demo_id):
        pass

        


