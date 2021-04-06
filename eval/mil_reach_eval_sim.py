import sys
sys.path.append("/home/caor/Documents/ur3/tecnets/reacher_env/gym-mil")
sys.path.append("/home/caor/Documents/ur3/tecnets/reacher_env/mujoco-py-0.5.7")
import gym
import imageio

import os
import numpy as np
import random
from torch.utils.data._utils.collate import default_collate

from eval.eval_sim import EvalSim

XML_PATH = '/home/caor/Documents/ur3/tecnets/reacher_env/gym-mil/gym/envs/mujoco/assets/sim_vision_reach_test_xmls/'
def path_to_xml(path):
    return XML_PATH + path.split('/')[-2] + '_cond_' + path[-7] + '.xml'

REACH_SUCCESS_THRESH = 0.05
REACH_SUCCESS_TIME = 10

class EvalSimMilReach(EvalSim):

    def __init__(self, dataset, num_subtasks, num_trials, support_size,
                 log_dir=".", record_gifs=False, render=True):
        self.env = gym.make('ReacherMILTest-v1')
        _ = self.env.reset()
        super().__init__(dataset, num_subtasks, num_trials, support_size,
                         log_dir, record_gifs, render)

    def evaluate(self, epoch, emb_mod, ctr_mod):
        print("Evaluating at iteration: %i" % epoch)
        successes = []
        subtask_indices = np.random.choice(len(self.dataset), self.num_subtasks, replace=False)
        for i in subtask_indices:
            num_demos = len(self.demos[i])
            if num_demos < self.support_size + 1:
                continue

            num_trials = min(num_demos, self.num_trials)
            support_demos = [random.sample(list(range(0, k)) + list(range(k+1, num_demos)), self.support_size)
                             for k in range(num_trials)]
            embeddings = self.get_embeddings(i, support_demos, emb_mod)

            for j in range(num_trials):
                _ = self.env.reset()
                self.env.env.next(path_to_xml(self.demos[i][j]['images_path']))
                if self.render:
                    self.env.render()
                distances = []
                observations = []
                for t in range(self.time_horizon):
                    if self.render:
                        self.env.render()
                    # Observation is shape (64,80,3)
                    obs, state = self.env.env.get_current_image_obs()
                    observations.append(obs)
                    obs = self.dataset.preprocess_image(obs)

                    action = self.get_action(obs, state, embeddings[j], ctr_mod)
                    ob, reward, done, reward_dict = self.env.step(np.squeeze(action))
                    dist = -reward_dict['reward_dist']
                    if t >= self.time_horizon - REACH_SUCCESS_TIME:
                        distances.append(dist)
                if np.amin(distances) <= REACH_SUCCESS_THRESH:
                    successes.append(1.)
                else:
                    successes.append(0.)
                self.save_gifs(observations, epoch, i, j)
                if self.render:
                    self.env.render(close=True)
                
        final_suc = np.mean(successes)
        print("Final success rate is %.5f" % final_suc)
        return final_suc

    def save_gifs(self, observations, epoch, subtask_id, demo_id):
        if self.record_gifs:
            gifs_dir = os.path.join(self.record_gifs_dir,
                'epoch_%i' % epoch, 'task_%d' % subtask_id)  
            if not os.path.exists(gifs_dir):
                os.makedirs(gifs_dir)
            record_gif_path = os.path.join(gifs_dir,
                'cond%d.samp0.gif' % int(self.demos[subtask_id][demo_id]['images_path'][-7]))
            video = np.array(observations)
            imageio.mimwrite(record_gif_path, video)