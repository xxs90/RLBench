import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import CloseDrawer
import matplotlib.pyplot as plt
import utils

class Agent(object):

    def __init__(self, action_shape):
        self.action_shape = action_shape

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)


env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
    obs_config=ObservationConfig(),
    headless=True,
    robot_setup='panda')
env.launch()

task = env.get_task(CloseDrawer)
agent = Agent(env.action_shape)

training_steps = 120
episode_length = 40
obs = None
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        descriptions, obs = task.reset()
        print(descriptions)
    action = agent.act(obs)
    print(action)
    obs, reward, terminate = task.step(action)

    # fig, ax = plt.subplots(1, 5, figsize=(15, 3))
    # ax[0].imshow(obs.front_depth)
    # ax[0].set_title('front')
    # ax[1].imshow(obs.overhead_depth)
    # ax[1].set_title('overhead')
    # ax[2].imshow(obs.wrist_depth)
    # ax[2].set_title('wrist')
    # ax[3].imshow(obs.left_shoulder_depth)
    # ax[3].set_title('left_shoulder')
    # ax[4].imshow(obs.right_shoulder_depth)
    # ax[4].set_title('right_shoulder')
    # plt.show()
    # pcd = o3d.geometry.PointCloud()
    # cloud_front = obs.front_point_cloud.reshape(-1, 3)
    # cloud_overhead = obs.overhead_point_cloud.reshape(-1, 3)
    # cloud_wrist = obs.front_point_cloud.reshape(-1, 3)
    # cloud_left_shoulder = obs.left_shoulder_point_cloud.reshape(-1, 3)
    # cloud_right_shoulder = obs.right_shoulder_point_cloud.reshape(-1, 3)
    # cloud = np.concatenate((cloud_front, cloud_overhead, cloud_wrist, cloud_left_shoulder, cloud_right_shoulder))
    # pcd.points = o3d.utility.Vector3dVector(cloud)
    # o3d.visualization.draw_geometries([pcd])

    cloud = utils.combinePointClouds(obs)
    depth_img = utils.getProjectImg(cloud, 1.0, 128, (0.35, 0.0, 1.2))
    plt.imshow(depth_img)
    plt.colorbar()
    plt.show()

print('Done')
env.shutdown()









