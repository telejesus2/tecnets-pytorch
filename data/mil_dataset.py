import numpy as np
import os
from natsort import natsorted
import pickle
from PIL import Image

from data.utils import DATASET_MIL as DATASET
from data.dataset import MetaDataset


class MILDataset(MetaDataset):

    def __init__(self, taskname, sequence_strategy, train_or_test='train',
                 time_horizon=50, img_shape=(64, 80, 3), state_dim=10, action_dim=2):
        super(MILDataset, self).__init__(taskname, sequence_strategy, train_or_test,
                 time_horizon, img_shape, state_dim, action_dim)
    
    def _load(self, taskname, train_or_test):
        subtasks = []
        task_path = os.path.join(DATASET, taskname, train_or_test)
        for subtask in natsorted(os.listdir(task_path)):
            subtask_path = os.path.join(task_path, subtask)
            if not os.path.isdir(subtask_path):
                continue
            pkl_file = subtask_path + '.pkl'
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
            sample_img_folders = natsorted(os.listdir(subtask_path))
            samples = []
            for e_idx, ex_file in enumerate(sample_img_folders):
                img_path = os.path.join(subtask_path, ex_file)
                x = {
                    'images_path': img_path,
                    'actions': data['actions'][e_idx],
                    'states': data['states'][e_idx]
                }
                # if 'demo_selection' in data:
                #     x['demo_selection'] = data['demo_selection']
                samples.append(x)
            subtasks.append(samples)
        return subtasks

    def _load_image(self, images_path, timestep):
        return np.array(Image.open(images_path + '/' + str(timestep) + '.gif').convert("RGB"))

    def preprocess_image(self, img):
        # [0, 255] to [-1, 1]
        return ((img.astype('float32') / 255.) * 2.) -1.
