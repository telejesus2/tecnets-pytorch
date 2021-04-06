import numpy as np
import os
import h5py
import time
from natsort import natsorted

from rlbench.backend.task import Task
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.tasks import ReachTarget

from data.rlbench_dataset import DATASET

HIGHDIM_ATTRIBUTES = [
    'front_depth',
    'front_mask',
    'front_rgb',  
    'wrist_depth', 
    'wrist_mask', 
    'wrist_rgb',
]

LOWDIM_ATTRIBUTES = [
    'gripper_joint_positions', 
    'gripper_matrix', 
    'gripper_open', 
    'gripper_pose', 
    'gripper_touch_forces', 
    'joint_forces', 
    'joint_positions', 
    'joint_velocities',
    'task_low_dim_state',
    'wrist_camera_matrix',
]


def collect_demos(env: Environment, task: Task, taskname, n_iter_per_var=50, n_demos_per_iter=1000):
    env.launch()
    task_env = env.get_task(task)

    for variation_index in range(1):#task_env.variation_count()):
        # set variation
        task_env.set_variation(variation_index)
        description, _ = task_env.reset()

        # collect and save demos
        start = time.time()
        for i in range(n_iter_per_var):
            np.random.seed(6)
            demos = task_env.get_demos(n_demos_per_iter, live_demos=True, max_attempts=1)
            demos = np.array(demos).flatten()
            print(demos.shape)
            save_to_hdf5(demos, taskname, description, variation_index)
        print("TIME: ", time.time() - start)

    # env.shutdown()
    return demos

def save_to_hdf5(demos, taskname, descriptions, variation_index=0):
    # save demos to files lowdim.hdf5 and highdim.hdf5
    # inside folder DATASET/taskname/variation_<variation_index>/sample_<sample_index>
    # all demos should belong to the same variation
    # TODO: add train test 

    # variation path
    variation_path = DATASET + taskname + '/variation_' + str(variation_index) + '/'
    if not os.path.exists(variation_path):
        os.makedirs(variation_path)

    with open(variation_path + "description.txt", "w") as text_file:
        for description in descriptions:
            text_file.write(description + '\n')

    # sample index
    i = len(os.listdir(variation_path))

    for demo in demos:

        # sample path
        sample_path = variation_path + 'sample_' + str(i) + '/'
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)

        with h5py.File(sample_path + 'lowdim.hdf5', "w") as f_low, h5py.File(sample_path + 'highdim.hdf5', "w") as f_high:
            obs = demo[0].__dict__
            attributes = obs.keys()
            for attr in attributes:
                if attr in LOWDIM_ATTRIBUTES:
                    if obs[attr] is None:
                        dset = f_low.create_dataset(attr, data=np.array([])) # to delete call: del f[attr]
                    else:
                        dset = f_low.create_dataset(attr, data=np.array([obs.__dict__[attr] for obs in demo]).astype('float32'))
                elif attr in HIGHDIM_ATTRIBUTES:
                    if obs[attr] is None:
                        dset = f_high.create_dataset(attr, data=np.array([])) # to delete call: del f[attr]
                    else:
                        dset = f_high.create_dataset(attr, data=np.array([obs.__dict__[attr] for obs in demo]).astype('float32'))
        i+=1


# obs_config = ObservationConfig()
# obs_config.front_camera.rgb = True
# obs_config.wrist_camera.rgb = True
# obs_config.joint_forces = False

# action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
# env = Environment(
#     action_mode, obs_config=obs_config, robot_configuration='ur3robotiq', headless=True)

# collect_demos(env, ReachTarget, "ReachTarget", n_iter_per_var=1, n_demos_per_iter=250)