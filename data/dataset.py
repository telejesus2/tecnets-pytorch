import numpy as np
import torch
from torch import default_generator
from torch._utils import _accumulate
from torch import randperm

VALID_SEQUENCE_STRATEGIES = ['first', 'last', 'first_last', 'all']

sequence_strategy_error = ('{} is not a valid sequence embedding strategy.')

class MetaDataset(torch.utils.data.Dataset):

    def __init__(self, taskname, sequence_strategy, train_or_test,
                 time_horizon, img_shape, state_dim, action_dim):
        """
        :param img_shape: (h, w, c)
        """   
        super().__init__()
        self.data = self._load(taskname, train_or_test)
        self.time_horizon = time_horizon
        self.img_shape = img_shape
        self.state_dim = state_dim   
        self.action_dim = action_dim

        self.sequence_strategy = sequence_strategy
        if sequence_strategy not in VALID_SEQUENCE_STRATEGIES:
            raise ValueError(sequence_strategy_error.format(sequence_strategy))
        self.frames = 1
        if sequence_strategy == 'first_last':
            self.frames = 2
        elif sequence_strategy == 'all':
            self.frames = self.time_horizon

    def __len__(self):
        return len(self.data)

    def len_subtask(self, index):
        return len(self.data[index])
    
    def _load(self, taskname, train_or_test):
        raise NotImplementedError

    def _load_image(self, images_path, timestep):
        pass
        # raise NotImplementedError

    def _preload_images(self, images_paths):
        pass

    def preprocess_image(self, img):
        # to [-1, 1]
        pass

    def postprocess_image(self, img):
        # [-1, 1] to [0, 1]
        return (img + 1) / 2.
 
    def __getitem__(self, index):
        subtask_index = index[0]
        sample_indices = index[1]
        ctrnet_timesteps = index[2]

        subtask = self.data[subtask_index]
        support_query_size = len(sample_indices)
        examples = len(ctrnet_timesteps)

        states = [subtask[i]['states'] for i in sample_indices]
        actions = [subtask[i]['actions'] for i in sample_indices]
        images_paths = [subtask[i]['images_path'] for i in sample_indices]
        # return [subtask[i]['images_path'][53:] for i in sample_indices]

        self._preload_images(images_paths)

        embnet_images, embnet_states, embnet_actions = [], [], []   
        for i in range(support_query_size):
            emb_images, emb_states, emb_actions = self._sequence_data(
                images_paths[i],
                states[i],
                actions[i])
            # images will be of shape (sequence, w, h, 3)
            embnet_images.append(emb_images)
            embnet_states.append(emb_states)
            embnet_actions.append(emb_actions)

        # Grab some random timesteps in one of the support and query trajectories
        # The first should be a support and the last should be a query
        s_ctrnet_timesteps = ctrnet_timesteps[: examples // 2]
        q_ctrnet_timesteps = ctrnet_timesteps[examples // 2 :]
        ctrnet_images = [self._load_image(images_paths[0], t) for t in s_ctrnet_timesteps]
        ctrnet_images += [self._load_image(images_paths[-1], t) for t in q_ctrnet_timesteps]
        ctrnet_states = [states[0][t] for t in s_ctrnet_timesteps]
        ctrnet_states += [states[-1][t] for t in q_ctrnet_timesteps]
        ctrnet_actions = [actions[0][t] for t in s_ctrnet_timesteps]
        ctrnet_actions += [actions[-1][t] for t in q_ctrnet_timesteps]

        return {
            'embnet_images': self.preprocess_image(np.array(embnet_images)),   # (support_query_size, frames, img_shape)
            'embnet_states': np.array(embnet_states),                           # (support_query_size, frames, state_dim)
            'embnet_actions': np.array(embnet_actions),                         # (support_query_size, frames, action_dim)
            'ctrnet_images': self.preprocess_image(np.array(ctrnet_images)),   # (examples_size, img_shape)
            'ctrnet_states': np.array(ctrnet_states),                           # (examples_size, state_dim)
            'ctrnet_actions': np.array(ctrnet_actions),                         # (examples_size, action_dim)
        }
    
    def _sequence_data(self, images_path, states, actions):
        if self.sequence_strategy == 'first':
            emb_images = [self._load_image(images_path, 0)]
            emb_states = [states[0]]
            emb_actions = [actions[0]]
        elif self.sequence_strategy == 'last':
            emb_images = [self._load_image(images_path, self.time_horizon - 1)]
            emb_states = [states[-1]]
            emb_actions = [actions[-1]]
        elif self.sequence_strategy == 'first_last':
            emb_images = [self._load_image(images_path, 0),
                          self._load_image(images_path, self.time_horizon - 1)]
            emb_states = [states[0], states[-1]]
            emb_actions = [actions[0], actions[-1]]
        elif self.sequence_strategy == 'all':
            emb_images = [self._load_image(images_path, t) for t in range(self.time_horizon)]
            emb_states = [states[t] for t in range(self.time_horizon)]
            emb_actions = [actions[t] for t in range(self.time_horizon)]
        else:
            raise ValueError(sequence_strategy_error.format(self.sequence_strategy))
        return emb_images, emb_states, emb_actions


class MetaSubset(torch.utils.data.Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        self.time_horizon = dataset.time_horizon
        self.img_shape = dataset.img_shape
        self.state_dim = dataset.state_dim 
        self.action_dim = dataset.action_dim
        self.frames = dataset.frames

    def __getitem__(self, idx):
        idx[0] = self.indices[idx[0]]
        return self.dataset[idx]
        
    def __len__(self):
        return len(self.indices)

    def len_subtask(self, index):
        return len(self.dataset.data[self.indices[index]])


def random_split(dataset, lengths, generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if sum(lengths) != len(dataset):  # type: ignore
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [MetaSubset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

