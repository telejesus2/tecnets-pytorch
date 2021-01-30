import os
import h5py
from natsort import natsorted

from data.utils import DATASET_RLBENCH as DATASET
from data.dataset import MetaDataset


class RLBenchDataset(MetaDataset):

    def __init__(self, taskname, sequence_strategy, train_or_test='train',
                 time_horizon=17, img_shape=(128, 128, 3), state_dim=6, action_dim=6):
        super(RLBenchDataset, self).__init__(taskname, sequence_strategy, train_or_test,
                 time_horizon, img_shape, state_dim, action_dim)

    def _load(self, taskname, train_or_test):
        subtasks = []
        task_path = os.path.join(DATASET, taskname, train_or_test)
        for subtask in natsorted(os.listdir(task_path)):     
            samples = []
            subtask_path = os.path.join(task_path, subtask)
            for sample in natsorted(os.listdir(subtask_path)):
                sample_path = os.path.join(subtask_path, sample)
                with h5py.File(sample_path + '/lowdim.hdf5') as f:
                    x = {
                        'images_path': sample_path,
                        'actions': self._build_action(f),
                        'states': self._build_state(f)
                    }
                samples.append(x)            
            subtasks.append(samples) 
        return subtasks 

    def _load_image(self, images_path, timestep):
        return self.images[images_path[-3:]][timestep]

    def _preload_images(self, images_paths):
        self.images = {}
        for images_path in images_paths:
            self.images[images_path[-3:]] = self._load_images(images_path)

    def _load_images(self, images_path):
        with h5py.File(images_path + '/highdim.hdf5') as f:
            return self._build_images(f) 

    def _build_state(self, hdf5_file):
        return hdf5_file['joint_velocities'][...]
        # for gname, group in hdf5_file.items():
        #         if gname in self.attributes:
        #             if self.transform is None:
        #                 x[gname] = group[...]   # x[gname] = torch.from_numpy(group[...])                       
        #             else:
        #                 x[gname] = self.transform[gname](group[...])

    def _build_action(self, hdf5_file):
        return hdf5_file['joint_velocities'][...]

    def _build_images(self, hdf5_file):
        return hdf5_file['front_rgb'][...]

    def preprocess_image(self, img):
        # [0, 1] to [-1, 1]
        return (img * 2.) - 1.


































       
    # def __getitem__(self, index):
    #     subtask_index = index[0]
    #     sample_indices = index[1:]

    #     subtask = self.data[subtask_index]
    #     support_query_size = len(sample_indices)
    #     states = [subtask[i]['states'] for i in sample_indices]
    #     actions = [subtask[i]['actions'] for i in sample_indices]
    #     images = [self._load_images(subtask[i]['images_path']) for i in sample_indices]
    #     # return [subtask[i]['images_path'][53:] for i in sample_indices]

    #     embnet_images, embnet_states, embnet_actions = [], [], []   
    #     for i in range(support_query_size):
    #         emb_images, emb_states, emb_actions = self._sequence_data(
    #             images[i],
    #             states[i],
    #             actions[i])
    #         # images will be of shape (sequence, w, h, 3)
    #         embnet_images.append(emb_images)
    #         embnet_states.append(emb_states)
    #         embnet_actions.append(emb_actions)

    #     # Grab a random timestep in one of the support and query trajectories
    #     ctrnet_timestep = np.random.randint(0, self.time_horizon, 2, np.int32)
    #     # The first should be a support and the last should be a query
    #     ctrnet_images = [images[0][ctrnet_timestep[0]],
    #                      images[-1][ctrnet_timestep[1]]]
    #     ctrnet_states = [states[0][ctrnet_timestep[0]],
    #                      states[-1][ctrnet_timestep[1]]]
    #     ctrnet_actions = [actions[0][ctrnet_timestep[0]],
    #                       actions[-1][ctrnet_timestep[1]]]

    #     return (self._preprocess(np.array(embnet_images)), np.array(embnet_states), np.array(embnet_actions),
    #             self._preprocess(np.array(ctrnet_images)), np.array(ctrnet_states), np.array(ctrnet_actions))
        
    #     return {
    #         'embnet_images': self._preprocess(np.array(embnet_images)),
    #         'embnet_states': np.array(embnet_states),
    #         'embnet_actions': np.array(embnet_actions),
    #         'ctrnet_images': self._preprocess(np.array(ctrnet_images)),
    #         'ctrnet_states': np.array(ctrnet_states),
    #         'ctrnet_actions': np.array(ctrnet_actions),
    #         # 'training': training,
    #         # 'support': tf.placeholder_with_default(self.support, None),
    #         # 'query': tf.placeholder_with_default(self.query, None),
    #     }
    
    # def _sequence_data(self, images, states, actions):
    #     if self.sequence_strategy == 'first':
    #         emb_images = [images[0]]
    #         emb_states = [states[0]]
    #         emb_actions = [actions[0]]
    #     elif self.sequence_strategy == 'last':
    #         emb_images = [images[-1]]
    #         emb_states = [states[-1]]
    #         emb_actions = [actions[-1]]
    #     elif self.sequence_strategy == 'first_last':
    #         emb_images = [images[0], images[-1]]
    #         emb_states = [states[0], states[-1]]
    #         emb_actions = [actions[0], actions[-1]]
    #     elif self.sequence_strategy == 'all':
    #         emb_images = images
    #         emb_states = states
    #         emb_actions = actions
    #     else:
    #         raise ValueError(sequence_strategy_error.format(self.sequence_strategy))
    #     return emb_images, emb_states, emb_actions


