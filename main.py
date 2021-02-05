import numpy as np
from datetime import datetime
import os
import time

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
from network.utils import padding_same, conv_shape
from eval.mil_reach_eval_sim import EvalSimMilReach
from eval.eval_emb import EvalEmbedding

if __name__ == "__main__":

    #============================================================================================#
    # Data
    #============================================================================================#

    train_size = 1500 #1500
    val_size = 128 #128
    batch_size = 256 #64 256
    support_size = 2 #2
    query_size = 2 #2
    examples_size = 10 #2

    # training_dataset = RLBenchDataset('ReachTarget', 'first_last')
    training_dataset = MILDataset('mil_sim_reach', 'first_last')
    train_dataset, val_dataset = random_split(training_dataset, [train_size, val_size])

    loader_params = {'batch_size': batch_size,
                    'replacement': False, 
                    'shuffle': True,
                    'num_workers': 8, #8
                    'prefetch_factor': 2,
                    'pin_memory': True}

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
    # Eval
    #============================================================================================#

    eval = True

    log_dir = './logs/' + datetime.now().strftime('%m%d-%H%M%S')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    test_dataset = MILDataset('mil_sim_reach', 'first_last', 'test')
    # test_dataset = RLBenchDataset('ReachTarget', 'first_last', 'test')

    eval_sim = EvalSimMilReach(test_dataset, 10, 10, support_size, log_dir=log_dir, record_gifs=True, render=True)
    eval_emb = EvalEmbedding(test_dataset, 10, 4)

    #============================================================================================#
    # Networks
    #============================================================================================#

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    emb_dim = 20
    filters = [40,40,40] 
    kernels = [3,3,3] 
    strides = [2,2,2]
    dilations = [1,1,1]
    # paddings = padding_same(data_sizes['img_shape'][:2], kernels, strides, dilations)
    paddings = [2,2,2]
    fc_layers = [200,200,200,200]
    margin = 0.1

    emb_mod = EmbeddingModule(data_sizes, emb_dim, filters, kernels, strides, paddings, dilations, fc_layers, device, margin)
    ctr_mod = ControlModule(data_sizes, emb_dim, filters, kernels, strides, paddings, dilations, fc_layers, device)

    #============================================================================================#
    # Optimizer and Learner
    #============================================================================================#

    lr = 0.0005
    params = list(emb_mod.parameters()) + list(ctr_mod.parameters())

    opt = optim.Adam(params, lr=lr) # TODO should i have one optimizer per network ?
    # scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[7, 30, 70], gamma=0.8)

    loss_params = {'lambda_embedding': 1.0,
                'lambda_support': 1.0, 
                'lambda_query': 1.0}

    meta_learner = MetaLearner(train_gen, val_gen, emb_mod, ctr_mod, opt, eval_sim, eval_emb, **loss_params)

    #============================================================================================#
    # Training
    #============================================================================================#

    epochs = 400000 # 400000
    resume_epoch = 41000 
    SAVE_INTERVAL = 1000
    EVAL_INTERVAL = 10000
    VAL_INTERVAL = 10

    if resume_epoch > 0:
        print("resuming...")
        meta_learner.resume('./logs/0203-001353', resume_epoch, device)

    train_writer = SummaryWriter("./runs3/_train")
    valid_writer = SummaryWriter("./runs3/_valid")
    emb_writer = SummaryWriter("./runs3/_embed")

    # meta_learner.meta_train(0, train_emb=False, writer=None, log_interval=10)
    # meta_learner.evaluate(resume_epoch, writer=emb_writer)

    start = time.time()
    for epoch in range(resume_epoch + 1, epochs + 1):
        #scheduler.step()
        print("# {}".format(epoch))
        meta_learner.meta_train(epoch, train_emb=True, train_ctr=True, writer=train_writer, log_interval=2)
        if epoch % VAL_INTERVAL == 0:
            meta_learner.meta_valid(epoch, train_emb=True, train_ctr=True, writer=valid_writer)
        if epoch % SAVE_INTERVAL == 0:
            meta_learner.save(log_dir, epoch)
        if epoch % EVAL_INTERVAL == 0 and eval:
            meta_learner.evaluate(epoch, writer=emb_writer)
        # if epoch % 10 == 0:
        #     print(time.time() - start)
        #     start = time.time()
        











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

