import torch
import torch.nn as nn
import numpy as np

import gym
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv
from q2_schedule import LinearExploration, LinearSchedule
from q3_linear_torch import Linear
from configs.q6_train_atari_dddqn import config


class DDDQN(Linear):
    """
    Implementing the Dueling Double DQN.
    """

    def initialize_models(self):

        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n
        strides = np.array([4, 2, 1])  # The stride size for every conv2d layer
        filter_sizes = np.array([8, 4, 3])  # The filter size for every conv2d layer
        numb_filters = np.array([32, 64, 64])  # number of filters for every conv2d layer

        # Set initial values for fcl, input_size and padding
        paddings = ((strides - 1) * img_height - strides + filter_sizes) // 2
        input_size = self.config.state_history * n_channels
        fcl_input_h = img_height
        fcl_input_w = img_width

        # Calculate input shape for fully connected layer.
        for i in range(3):
            fcl_input_h = 1 + ((fcl_input_h + (paddings[i] * 2) - filter_sizes[i]) // strides[i])
            fcl_input_w = 1 + ((fcl_input_w + (paddings[i] * 2) - filter_sizes[i]) // strides[i])

        self.q_network = nn.Sequential(
            nn.Conv2d(input_size, numb_filters[0], filter_sizes[0], strides[0], paddings[0]),
            nn.ReLU(),
            nn.Conv2d(numb_filters[0], numb_filters[1], filter_sizes[1], strides[1], paddings[1]),
            nn.ReLU(),
            nn.Conv2d(numb_filters[1], numb_filters[2], filter_sizes[2], strides[2], paddings[2]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fcl_input_h * fcl_input_w * numb_filters[2], 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        self.target_network = nn.Sequential(
            nn.Conv2d(input_size, numb_filters[0], filter_sizes[0], strides[0], paddings[0]),
            nn.ReLU(),
            nn.Conv2d(numb_filters[0], numb_filters[1], filter_sizes[1], strides[1], paddings[1]),
            nn.ReLU(),
            nn.Conv2d(numb_filters[1], numb_filters[2], filter_sizes[2], strides[2], paddings[2]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fcl_input_h * fcl_input_w * numb_filters[2], 512),
            nn.ReLU()
        )

        self.SV = nn.Linear(512, 1).to(self.device)
        self.AV = nn.Linear(512, num_actions).to(self.device)

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions
        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"
        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)
        """

        input_state_arg = state.permute(0, 3, 1, 2)
        out = self.q_network(input_state_arg) if network == 'q_network' else self.target_network(input_state_arg)

        # Get Q value using the state-value function and the action value.
        out = self.SV(out) + (self.AV(out) - torch.mean(self.AV(out), dim=1, keepdim=True))

        return out


"""
Use Dueling Double deep Q network for test environment.
"""
if __name__ == '__main__':
    # make env
    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
                        overwrite_render=config.overwrite_render)

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = DDDQN(env, config)
    model.run(exp_schedule, lr_schedule)