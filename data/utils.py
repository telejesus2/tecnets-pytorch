import numpy as np
import os
import h5py
import time
from natsort import natsorted

# from rlbench.backend.task import Task
# from rlbench.environment import Environment

DATASET_RLBENCH = '/home/caor/Documents/datasets/demos/'
DATASET_MIL = '/home/caor/Documents/datasets/tecnets/'

HIGHDIM_ATTRIBUTES = ['front_depth',
                      'front_mask',
                      'front_rgb', 
                      'left_shoulder_depth', 
                      'left_shoulder_mask', 
                      'left_shoulder_rgb', 
                      'right_shoulder_depth', 
                      'right_shoulder_mask', 
                      'right_shoulder_rgb', 
                      'wrist_camera_matrix', 
                      'wrist_depth', 
                      'wrist_mask', 
                      'wrist_rgb']

LOWDIM_ATTRIBUTES = ['gripper_joint_positions', 
                     'gripper_matrix', 
                     'gripper_open', 
                     'gripper_pose', 
                     'gripper_touch_forces', 
                     'joint_forces', 
                     'joint_positions', 
                     'joint_velocities',
                     'task_low_dim_state']

# def collect_demos(env: Environment, task: Task, taskname, n_loops=50):
#     # TODO: add subtask index, or add file with subtask info (e.g. go to the red)

#     env.launch()
#     task_env = env.get_task(task)
#     start = time.time()
#     for i in range(n_loops):
#         demos = task_env.get_demos(1000, live_demos=True, max_attempts=1)
#         demos = np.array(demos).flatten()
#         save_to_hdf5(demos, taskname)
#     print("TIME: ", time.time() - start)
#     env.shutdown()

def save_to_hdf5(demos, taskname, subtask_index=0):
    # save demos to files lowdim.hdf5 and highdim.hdf5
    # inside folder DATASET/taskname/subtask_<subtask_index>/sample_<sample_index>
    # all demos should belong to the same subtask
    # TODO: add train test 

    subtask_path = DATASET_RLBENCH + taskname + '/subtask_' + str(subtask_index) + '/'
    if not os.path.exists(subtask_path):
        os.makedirs(subtask_path)
    i = len(os.listdir(subtask_path))

    for demo in demos:

        sample_path = subtask_path + 'sample_' + str(i) + '/'
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

