import numpy as np

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from rlbench.backend.observation import Observation    

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data.utils import save_to_hdf5, get_files
from data.rlbench_dataset import RLBenchDataset
from data.dataset import random_split
from data.mil_dataset import MILDataset
from data.dataloader import MetaDataLoader
from network.embedding import EmbeddingNet
from network.control import ControlNet 
from meta_learner import MetaLearner

#============================================================================================#
# Data
#============================================================================================#

train_size = 1500 #1500
val_size = 150 #150
batch_size = 1 #64
support_size = 2 #2
query_size = 2 #2
num_ctr_timesteps = 2

# training_dataset = RLBenchDataset('ReachTarget', 'first_last')
training_dataset = MILDataset('mil_sim_reach', 'first_last')
train_dataset, val_dataset = random_split(training_dataset, [train_size, val_size])

params = {'batch_size': batch_size,
          'replacement': False, 
          'shuffle': False,
          'num_workers': 0}

train_gen = MetaDataLoader(train_dataset, support_size, query_size, num_ctr_timesteps, **params)
val_gen = MetaDataLoader(val_dataset, support_size, query_size, num_ctr_timesteps, **params)

#============================================================================================#
# Networks
#============================================================================================#

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

emb_dim = 20
filters = [40,40,40] 
kernels = [3,3,3] 
strides = [2,2,2]
paddings = [2, 2, 2] # TODO es 'same'
fc_layers = [200,200,200,200] 

emb_net = ControlNet(training_dataset.img_shape, training_dataset.state_dim, training_dataset.action_dim, emb_dim,
                     filters, kernels, strides, paddings, fc_layers,
                     norm_conv='layer', norm_fc=None, act='elu', drop_rate_conv=0.0, drop_rate_fc=0.0)

ctr_net = EmbeddingNet(training_dataset.img_shape, emb_dim,
                       filters, kernels, strides, paddings, fc_layers,
                       norm_conv='layer', norm_fc=None, act='elu', drop_rate_conv=0.0, drop_rate_fc=0.0)

emb_net = emb_net.to(device)
ctr_net = ctr_net.to(device)
# emb_net = torch.nn.DataParallel(emb_net, device_ids=[0])
# ctr_net = torch.nn.DataParallel(ctr_net, device_ids=[0])
# print('Number of parameters: {}'.format(get_n_params(model)))

params = list(emb_net.parameters()) + list(ctr_net.parameters())

print("emb_net params")
for name, param in emb_net.named_parameters():
    print(name, param.shape)

print("ctr_net params")
for name, param in ctr_net.named_parameters():
    print(name, param.shape)

#============================================================================================#
# Training
#============================================================================================#

epochs = 400000
lr = 0.0005

opt = optim.Adam(params, lr=lr)

params = {'lambda_embedding': 1.0,
          'lambda_support': 0.1, 
          'lambda_query': 0.1,
          'margin': 0.1}

meta_learner = MetaLearner(emb_net, ctr_net, train_gen, val_gen, opt, device, **params)

for epoch in range(epochs):
    print("# {}".format(epoch+1))
    meta_learner.meta_train(epoch)
    meta_learner.meta_valid(epoch)


# ACTION_MODE = [ ArmActionMode.ABS_JOINT_VELOCITY,
#                 ArmActionMode.DELTA_JOINT_VELOCITY,
#                 ArmActionMode.ABS_JOINT_POSITION,
#                 ArmActionMode.DELTA_JOINT_POSITION,
#                 ArmActionMode.ABS_JOINT_TORQUE,
#                 ArmActionMode.DELTA_JOINT_TORQUE,
#                 ArmActionMode.ABS_EE_POSE_WORLD_FRAME,
#                 ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME,
#                 ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME,
#                 ArmActionMode.DELTA_EE_POSE_WORLD_FRAME,
#                 ArmActionMode.EE_POSE_EE_FRAME,
#                 ArmActionMode.EE_POSE_PLAN_EE_FRAME ]

# obs_config = ObservationConfig()
# obs_config.set_all_high_dim(False)
# obs_config.front_camera.rgb = True
# action_mode = ActionMode(ACTION_MODE[0])
# env = Environment(
#     action_mode, '', obs_config, headless=True,
#     robot_configuration='ur3')

# tasks = ['ReachTarget', 'CloseDrawer']
# tasks = ['CloseBox']
# attributes = ['front_rgb', 'joint_velocities', 'joint_positions']



