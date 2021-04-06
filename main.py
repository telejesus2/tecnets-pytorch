import numpy as np
from datetime import datetime
import os
import time
from numpy.lib.npyio import load   

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
from eval.mil_reach_eval_sim import EvalSimMilReach
from eval.eval_emb import EvalEmbedding
from utils import save_params

def train(
    n_iter,
    resume_iter,
    seed,
    logdir,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    main_params = {
        'n_iter': n_iter,
        'resume_iter': resume_iter,
        'seed': seed}

    #============================================================================================#
    # Data
    #============================================================================================#

    # train_size = 1500 #1500
    # val_size = 128 #128
    # batch_size = 256 #64
    # support_size = 2 #2
    # query_size = 2 #2
    # examples_size = 10 #2

    train_size = 16 #1500
    val_size = 4 #128
    batch_size = 16 #64
    support_size = 2 #2
    query_size = 2 #2
    examples_size = 10 #2

    training_dataset = RLBenchDataset('ReachTarget', 'first_last', control=False)
    # training_dataset = MILDataset('mil_sim_reach', 'first_last')
    train_dataset, val_dataset = random_split(training_dataset, [train_size, val_size])

    loader_params = {
        'batch_size': batch_size,
        'replacement': False, 
        'shuffle': True,
        'num_workers': 8, #8
        'prefetch_factor': 2,
        'pin_memory': True}

    train_gen = MetaDataLoader(train_dataset, support_size, query_size, examples_size, **loader_params)
    val_gen = MetaDataLoader(val_dataset, support_size, query_size, examples_size, **loader_params)

# ###
# a = next(iter(train_gen))
# im = a['embnet_images']
# print(im.shape)


    data_params = {
        'support_size': train_gen.support_size,
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

    test_dataset = MILDataset('mil_sim_reach', 'first_last', 'test')
    # test_dataset = RLBenchDataset('ReachTarget', 'first_last', 'test')

    eval_sim = EvalSimMilReach(test_dataset, 10, 10, support_size, log_dir=logdir, record_gifs=True, render=True)
    eval_emb = EvalEmbedding(val_dataset, 4, 10)

    #============================================================================================#
    # Networks
    #============================================================================================#

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net_params = {
        'emb_dim': 20,
        'filters': [40,40,40],
        'kernels': [3,3,3],
        'strides': [2,2,2],
        'dilations': [1,1,1],
        'paddings': [2,2,2],
        'fc_layers': [200,200],
        'margin': 0.1}
    # net_params['paddings'] = padding_same(data_sizes['img_shape'][:2], net_params['kernels'], net_params['strides'], net_params['dilations'])

    emb_mod = EmbeddingModule(data_params, net_params['emb_dim'], net_params['filters'], net_params['kernels'],
        net_params['strides'], net_params['paddings'], net_params['dilations'], net_params['fc_layers'], device,
        net_params['margin'])
    ctr_mod = None #ControlModule(data_params, net_params['emb_dim'], net_params['filters'], net_params['kernels'],
        # net_params['strides'], net_params['paddings'], net_params['dilations'], net_params['fc_layers'], device)

    #============================================================================================#
    # Optimizer and Learner
    #============================================================================================#

    opt_params = {
        'ctr_lr': 0.0005,
        'emb_lr': 0.0005}
    
    # params = list(emb_mod.parameters()) + list(ctr_mod.parameters())
    # opt = optim.Adam(params, lr=lr) # TODO should i have one optimizer per network ?
    opt_ctr = None #optim.Adam(ctr_mod.parameters(), lr=opt_params['ctr_lr'])
    opt_emb = optim.Adam(emb_mod.parameters(), lr=opt_params['emb_lr']) 

    # scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[7, 30, 70], gamma=0.8)

    train_params = {
        'lambda_embedding': 1.0,
        'lambda_support': 10, 
        'lambda_query': 10,
        'train_emb': True,
        'train_ctr': False}

    meta_learner = MetaLearner(train_gen, val_gen, emb_mod, opt_emb, ctr_mod, opt_ctr, eval_sim, eval_emb, **train_params)

    #============================================================================================#
    # Logger
    #============================================================================================#

    save_params(logdir, {**main_params, **data_params, **loader_params, **net_params, **opt_params, **train_params})

    train_writer = SummaryWriter(os.path.join(logdir, 'runs/_train'))
    valid_writer = SummaryWriter(os.path.join(logdir, 'runs/_valid'))
    eval_writer = SummaryWriter(os.path.join(logdir, 'runs/_eval'))

    #============================================================================================#
    # Training
    #============================================================================================#

    epochs = n_iter # 400000
    resume_epoch = resume_iter 
    SAVE_INTERVAL = 5000
    EVAL_INTERVAL = 500
    VAL_INTERVAL = 50
    LOG_INTERVAL = 50

    if resume_epoch > 0:
        print("resuming...")
        # meta_learner.resume(logdir, resume_epoch, device)
        meta_learner.resume('logs/ctr_18-02-2021_23-37-37/5', resume_epoch, device)

    # meta_learner.meta_train(0, train_emb=False, writer=None, log_interval=10)
    # meta_learner.evaluate(resume_epoch, writer=eval_writer)

    start = time.time()
    for epoch in range(resume_epoch + 1, epochs + 1):
        #scheduler.step()
        print("# {}".format(epoch))
        meta_learner.meta_train(epoch, writer=train_writer, log_interval=2, log_histograms=(epoch%LOG_INTERVAL==0))
        if epoch % VAL_INTERVAL == 0:
            meta_learner.meta_valid(epoch, writer=valid_writer)
        if epoch % EVAL_INTERVAL == 0 and eval:
            meta_learner.evaluate(epoch, writer=eval_writer)
        if epoch % SAVE_INTERVAL == 0:
            meta_learner.save(logdir, epoch)
        # if epoch % 10 == 0:
        #     print(time.time() - start)
        #     start = time.time()
        
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='all')
    parser.add_argument('--n_iter', '-ni', type=int, default=200000)
    parser.add_argument('--resume_iter', '-ri', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    args = parser.parse_args()

    if not(os.path.exists('logs')):
        os.makedirs('logs')
    logdir = args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('logs', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)

        train(
            args.n_iter,
            args.resume_iter,
            seed,
            os.path.join(logdir,'%d'%seed),
        )

if __name__ == "__main__":
    main()


# tasks = ['ReachTarget', 'CloseDrawer']
# tasks = ['CloseBox']
# attributes = ['front_rgb', 'joint_velocities', 'joint_positions']