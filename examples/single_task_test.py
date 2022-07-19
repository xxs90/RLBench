import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import PutItemInDrawer as rltasks
import matplotlib.pyplot as plt
import utils
from pyrep.robots.robot_component import RobotComponent as rc


class Agent(object):

    def __init__(self, action_shape):
        self.action_shape = action_shape

    def ingest(self, demos):
        pass

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
        # # arm = [-4.0, -2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        # print(arm)

        # arm = np.random.normal(1.0, 1.0, size=(self.action_shape[0] - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)

live_demos = True
obs_config = ObservationConfig()
obs_config.set_all(True)

# action_mode = MoveArmThenGripper(arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=Discrete())
action_mode = MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())

env = Environment(
    action_mode=action_mode, obs_config=ObservationConfig(), headless=False, robot_setup='panda')
env.launch()

task = env.get_task(rltasks)
demos = task.get_demos(1, live_demos=True)

agent = Agent(env.action_shape)
agent.ingest(demos)
# rc.get_joint_positions(agent)

training_steps = 20
episode_length = 4
obs = None
# x1 = 0  # -30.0 (lr)
# x2 = 0.3  # 30.0
# y1 = 0  # -30.0
# y2 = 0.0  # 30.0
# z1 = 1.2
# z2 = 1.4
# q_i = 0.0
# q_j = 1.0
# q_k = 0.0
# q = 0.0
# action = [[x1, y1, z1, q_i, q_j, q_k, q, 1.0], [x2, y2, z2, q_i, q_j, q_k, q, 1.0],
#           [x1, y1, z1, q_i, q_j, q_k, q, 1.0], [x2, y2, z2, q_i, q_j, q_k, q, 1.0],
#           [x1, y1, z1, q_i, q_j, q_k, q, 1.0], [x2, y2, z2, q_i, q_j, q_k, q, 1.0],
#           [x1, y1, z1, q_i, q_j, q_k, q, 1.0], [x2, y2, z2, q_i, q_j, q_k, q, 1.0],
#           [x1, y1, z1, q_i, q_j, q_k, q, 1.0], [x2, y2, z2, q_i, q_j, q_k, q, 1.0]]
# action = [x2, y2, z2, q_i, q_j, q_k, q, 1.0]
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

    # cloud = utils.combinePointClouds(obs)
    # depth_img = utils.getProjectImg(cloud, 1.0, 128, (0.35, 0.0, 1.2))
    # plt.imshow(depth_img)
    # plt.colorbar()
    # plt.show()

print('Done')
env.shutdown()
