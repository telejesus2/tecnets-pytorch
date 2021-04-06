
import numpy as np
from natsort import natsorted
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.tasks import ReachTarget

from data.rlbench_data_collection import collect_demos

obs_config = ObservationConfig()
obs_config.front_camera.rgb = True
obs_config.wrist_camera.rgb = True
obs_config.joint_forces = False

action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_WORLD_FRAME)
# action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)
# action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env = Environment(
    action_mode, obs_config=obs_config, robot_configuration='ur3robotiq', headless=False)
import time

time.sleep(3)

demos = collect_demos(env, ReachTarget, "hola", n_iter_per_var=1, n_demos_per_iter=3)

time.sleep(3)
#####################################

# env.launch()
task = env.get_task(ReachTarget)

np.random.seed(6)
for i in range(len(demos)):
    demo = demos[i]
    print('Reset Episode')
    descriptions, obs = task.reset()

    t = 0
    terminate = False
    while not terminate: 
        action = demo[t+1].gripper_pose
        # action = demo[t+1].joint_positions
        # action = demo[t+1].joint_velocities
        action = np.concatenate([action, [1.0]], axis=-1)
        obs, reward, terminate = task.step(action)
        t += 1

print('Done')
env.shutdown()