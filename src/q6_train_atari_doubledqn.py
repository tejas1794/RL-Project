import gym
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

import tensorflow as tf

from q2_schedule import LinearExploration, LinearSchedule
from q4_nature_torch import NatureQN
from configs.q6_train_atari_doubledqn import config


class DoubleDQN(NatureQN):
    """
    Implementation for Double DQN
    """
    def add_loss_op(self, q, target_q):
        num_actions = self.env.action_space.n
        y_val = self.r + self.config.gamma * tf.reduce_sum(tf.multiply(target_q, tf.one_hot(tf.arg_max(q, dimension=1), num_actions)), axis=1)
        q_sample = tf.where(self.done_mask, self.r, y_val)
        q_new = tf.reduce_sum(tf.multiply(tf.one_hot(self.a, num_actions), q), axis=1)
        self.loss = tf.reduce_mean(tf.square(q_new - q_sample))


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
    model = DoubleDQN(env, config)
    model.run(exp_schedule, lr_schedule)