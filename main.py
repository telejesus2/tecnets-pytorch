import numpy as np
from datetime import datetime
import os

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from rlbench.backend.observation import Observation    

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data.rlbench_dataset import RLBenchDataset
from data.mil_dataset import MILDataset
from data.dataset import random_split
from data.dataloader import MetaDataLoader
from embedding_module import EmbeddingModule
from control_module import ControlModule
from meta_learner import MetaLearner
from network.utils import padding_same


#============================================================================================#
# Data
#============================================================================================#

train_size = 1500 #1500
val_size = 128 #128
batch_size = 64 #64
support_size = 2 #2
query_size = 2 #2
examples_size = 2

# training_dataset = RLBenchDataset('ReachTarget', 'first_last')
training_dataset = MILDataset('mil_sim_reach', 'first_last')
train_dataset, val_dataset = random_split(training_dataset, [train_size, val_size])

loader_params = {'batch_size': batch_size,
                 'replacement': False, 
                 'shuffle': True,
                 'num_workers': 0}

train_gen = MetaDataLoader(train_dataset, support_size, query_size, examples_size, **loader_params)
val_gen = MetaDataLoader(val_dataset, support_size, query_size, examples_size, **loader_params)

data_sizes = {'support_size': train_gen.support_size,
              'query_size': train_gen.query_size, 
              'examples_size': train_gen.examples_size,
              'img_shape': train_dataset.img_shape,
              'time_horizon': train_dataset.time_horizon,
              'img_shape': train_dataset.img_shape,
              'state_dim': train_dataset.state_dim,
              'action_dim': train_dataset.action_dim,
              'frames': train_dataset.frames}

#============================================================================================#
# Networks
#============================================================================================#

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

emb_dim = 20
filters = [40,40,40] 
kernels = [3,3,3] 
strides = [2,2,2]
dilations = [1, 1, 1]
paddings = padding_same(data_sizes['img_shape'][:2], kernels, strides, dilations)
fc_layers = [200,200,200,200]
margin = 0.1

emb_mod = EmbeddingModule(data_sizes, emb_dim, filters, kernels, strides, paddings, dilations, fc_layers, device, margin)
ctr_mod = ControlModule(data_sizes, emb_dim, filters, kernels, strides, paddings, dilations, fc_layers, device)

#============================================================================================#
# Eval
#============================================================================================#

#============================================================================================#
# Optimizer and Learner
#============================================================================================#

lr = 0.0005
params = list(emb_mod.parameters()) + list(ctr_mod.parameters())

opt = optim.Adam(params, lr=lr) # TODO should i have one optimizer per network ?
# scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[7, 30, 70], gamma=0.8)

loss_params = {'lambda_embedding': 1.0,
               'lambda_support': 0.1, 
               'lambda_query': 0.1}

meta_learner = MetaLearner(train_gen, val_gen, emb_mod, ctr_mod, opt, **loss_params)

#============================================================================================#
# Training
#============================================================================================#

epochs = 100 # 400000
resume_epoch = 0
SAVE_INTERVAL = 10
EVAL_INTERVAL = 20

if resume_epoch > 0:
    print("resuming...")
    meta_learner.resume('./logs/0126-190858', resume_epoch, device)

log_dir = "./logs/" + datetime.now().strftime('%m%d-%H%M%S') 
print(log_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

train_writer = SummaryWriter("./runs/_train")
valid_writer = SummaryWriter("./runs/_valid")

for epoch in range(resume_epoch, epochs):
    #scheduler.step()
    print("# {}".format(epoch))
    meta_learner.meta_train(epoch, writer=train_writer)
    meta_learner.meta_valid(epoch, writer=valid_writer)
    if epoch % SAVE_INTERVAL == 0:
        meta_learner.save(log_dir, epoch)
    if epoch % EVAL_INTERVAL == 0 and epoch > 1 and eval is not None:
        acc = self.eval.evaluate(itr)
        print('Evaluation at iter %d. Success rate: %.2f' % (itr, acc))
        if self.summary_writer is not None:
            eval_success = sess.run(
                self.eval_summary, {self.eval_summary_in: acc})
            self.summary_writer.add_summary(sess, eval_success, itr)
        


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





test_dataset = MILDataset('mil_sim_reach', 'first_last', 'test')
loader_params = {'batch_size': len(test_dataset),
                 'replacement': False, 
                 'shuffle': False,
                 'num_workers': 0}
test_gen = MetaDataLoader(test_dataset, support_size, query_size, examples_size, **loader_params)
meta_learner.resume('./logs/0126-190858', 20, device)
meta_learner.emb_mod.eval()

#####

inputs = next(iter(test_gen))
embnet_images = inputs['embnet_images'].to(device) # (N, U_n + q_n, frames, img_shape) where img_shape = (h, w, c)

label_img = embnet_images[:,:,1,:]
label_img = label_img.view([-1] + list(label_img.shape[2:]))
label_img = label_img.permute(0,3,1,2)
label_img = (label_img +1) / 2.
print(label_img.shape)

embnet_images = torch.cat(torch.unbind(embnet_images, dim=2), dim=-1) # (N, U_n + q_n, h, w, c * frames)
embnet_images = embnet_images.view([-1] + list(embnet_images.shape[2:])) # (N * (U_n + q_n), h, w, c * frames)
sentences = meta_learner.emb_mod.emb_net(embnet_images) # (N * (U_n + q_n), emb_dim)
print(sentences.shape)

meta = []
for i in range(len(test_dataset)):
    meta += [i] * (support_size + query_size)
print(len(meta))

x = 20
writer = SummaryWriter("./caca")
writer.add_embedding(sentences[:x], metadata=meta[:x], label_img=label_img[:x])