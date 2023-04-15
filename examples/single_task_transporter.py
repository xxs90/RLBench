import os
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget as tasks
from models.attention import Attention
from models.transport import Transport
from models.transport_ablation import TransportPerPixelLoss
from models.transport_goal import TransportGoal
import tensorflow as tf
import matplotlib.pyplot as plt
from pyrep.robots.robot_component import RobotComponent as rc
from utils import utils
class TransporterAgent(object):

    def __init__(self, name, task, root_dir, n_rotations=36):
        self.name = name
        self.task = task
        self.total_steps = 0
        self.crop_size = 64
        self.n_rotations = n_rotations
        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.cam_config = ObservationConfig()
        self.models_dir = os.path.join(root_dir, 'checkpoints', self.name)
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])


    def ingest(self, demos):
        pass

    def get_image(self, obs):
        """Stack color and height images image."""

        # if self.use_goal_image:
        #   colormap_g, heightmap_g = utils.get_fused_heightmap(goal, configs)
        #   goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
        #   input_image = np.concatenate((input_image, goal_image), axis=2)
        #   assert input_image.shape[2] == 12, input_image.shape

        # Get color and height maps from RGB-D images.
        cmap, hmap = utils.get_fused_heightmap(
            obs, self.cam_config, self.bounds, self.pix_size)
        img = np.concatenate((cmap,
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None]), axis=2)
        assert img.shape == self.in_shape, img.shape
        return img

    def get_sample(self, dataset, augment=True):
        """Get a dataset sample.
        Args:
          dataset: a ravens.Dataset (train or validation)
          augment: if True, perform data augmentation.
        Returns:
          tuple of data for training:
            (input_image, p0, p0_theta, p1, p1_theta)
          tuple additionally includes (z, roll, pitch) if self.six_dof
          if self.use_goal_image, then the goal image is stacked with the
          current image in `input_image`. If splitting up current and goal
          images is desired, it should be done outside this method.
        """

        (obs, act, _, _), _ = dataset.sample()
        img = self.get_image(obs)

        # Get training labels from data sample.
        p0_xyz, p0_xyzw = act['pose0']
        p1_xyz, p1_xyzw = act['pose1']
        p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
        p0_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
        p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)
        p1_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
        p1_theta = p1_theta - p0_theta
        p0_theta = 0

        # Data augmentation.
        if augment:
            img, _, (p0, p1), _ = utils.perturb(img, [p0, p1])

        return img, p0, p0_theta, p1, p1_theta

    def train(self, dataset, writer=None):
        """Train on a dataset sample for 1 iteration.
        Args:
          dataset: a ravens.Dataset.
          writer: a TF summary writer (for tensorboard).
        """
        tf.keras.backend.set_learning_phase(1)
        img, p0, p0_theta, p1, p1_theta = self.get_sample(dataset)

        # Get training losses.
        step = self.total_steps + 1
        loss0 = self.attention.train(img, p0, p0_theta)
        if isinstance(self.transport, Attention):
            loss1 = self.transport.train(img, p1, p1_theta)
        else:
            loss1 = self.transport.train(img, p0, p1, p1_theta)
        with writer.as_default():
            sc = tf.summary.scalar
            sc('train_loss/attention', loss0, step)
            sc('train_loss/transport', loss1, step)
        print(f'Train Iter: {step} Loss: {loss0:.4f} {loss1:.4f}')
        self.total_steps = step

    def validate(self, dataset, writer=None):  # pylint: disable=unused-argument
        """Test on a validation dataset for 10 iterations."""
        print('Skipping validation.')

    def act(self, obs, info=None, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        tf.keras.backend.set_learning_phase(0)

        # Get heightmap from RGB-D images.
        img = self.get_image(obs)

        # Attention model forward pass.
        pick_conf = self.attention.forward(img)
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Transport model forward pass.
        place_conf = self.transport.forward(img, p0_pix)
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        # Pixels to end effector poses.
        hmap = img[:, :, 3]
        p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw))
        }

class OriginalTransporterAgent(TransporterAgent):

  def __init__(self, name, task, n_rotations=36):
    super().__init__(name, task, n_rotations)

    self.attention = Attention(
        in_shape=self.in_shape,
        n_rotations=1,
        preprocess=utils.preprocess)
    self.transport = Transport(
        in_shape=self.in_shape,
        n_rotations=self.n_rotations,
        crop_size=self.crop_size,
        preprocess=utils.preprocess)

# live_demos = True
# obs_config = ObservationConfig()
# obs_config.set_all(True)
#
action_mode = MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
#
env = Environment(
    action_mode=action_mode,
    obs_config=ObservationConfig(),
    headless=False,
    robot_setup='panda')
env.launch()
#
task = env.get_task(tasks)
demos = task.get_demos(2, live_demos=True)
agent = TransporterAgent(env.action_shape)
#
training_steps = 120
episode_length = 40
obs = None
#
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        descriptions, obs = task.reset()
        print(descriptions)
    action = agent.act(obs)
    print(action)
    obs, reward, terminate = task.step(action)
#
print('Done')
env.shutdown()
