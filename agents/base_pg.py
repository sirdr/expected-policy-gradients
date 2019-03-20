import os
import argparse
import sys
import logging
import time
import numpy as np
from scipy import integrate
from copy import copy

import tensorflow as tf

import gym
import scipy.signal
import os
import time
import inspect
from utils.general import get_logger, Progbar, export_plot
from config import get_config


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)



class PG(object):
    """
    Abstract Class for implementing a Policy Gradient Based Algorithm
    """
    def __init__(self, env, config, run=0, logger=None):
        """
        Initialize Policy Gradient Class

        Args:
                env: an OpenAI Gym environment
                config: class with hyperparameters
                logger: logger instance from the logging module

        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # store hyperparameters
        self.config = config
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)
        self.env = env

        # discrete vs continuous action space
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_shape = self.env.observation_space.shape
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]

        self.actor_lr = self.config.learning_rate
        self.critic_lr = self.config.critic_learning_rate
        self.tau = self.config.target_update_weight
        self.gamma = self.config.gamma

        if not self.discrete:
            self.action_high = float(self.env.action_space.high[0])
            self.action_low = float(self.env.action_space.low[0])
            self.obs_high = self.env.observation_space.high
            self.obs_low = self.env.observation_space.low

        self.agent_name = "pg"
        self.run = run


    def add_placeholders_op(self):
        """
        Add necessary placeholders for policy gradient algorithm
        """
        raise NotImplementedError()


    def add_actor_loss_op(self):
        """
        Compute the loss, averaged for a given batch.

        """
        raise NotImplementedError()

    def add_actor_optimizer_op(self):
        """
        Set 'self.train_op' using AdamOptimizer
        """
        raise NotImplementedError()

    def build(self):
        """
        Build the model by adding all necessary variables.
        """
        raise NotImplementedError()

    def initialize(self):
        # create tf session
        self.sess = tf.Session()

        self.add_summary()
        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init)

    def add_summary(self):
        """
        Tensorboard stuff.

        You don't have to change or use anything here.
        """
        # extra placeholders to log stuff from python
        self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
        self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

        self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
        tf.summary.scalar("Max Reward", self.max_reward_placeholder)
        tf.summary.scalar("Std Reward", self.std_reward_placeholder)
        tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)

        # logging
        self.merged = tf.summary.merge_all()
        # self.file_writer = tf.summary.FileWriter(self.config.output_path,self.sess.graph)

    def init_averages(self):
        """
        Defines extra attributes for tensorboard.

        You don't have to change or use anything here.
        """
        self.avg_reward = 0.
        self.max_reward = 0.
        self.std_reward = 0.
        self.eval_reward = 0.

    def update_averages(self, rewards, scores_eval):
        """
        Update the averages.

        You don't have to change or use anything here.

        Args:
            rewards: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        if len(scores_eval) > 0:
          self.eval_reward = scores_eval[-1]

    def record_summary(self, t):
        """
        Add summary to tensorboard

        You don't have to change or use anything here.
        """

        fd = {
          self.avg_reward_placeholder: self.avg_reward,
          self.max_reward_placeholder: self.max_reward,
          self.std_reward_placeholder: self.std_reward,
          self.eval_reward_placeholder: self.eval_reward,
        }
        summary = self.sess.run(self.merged, feed_dict=fd)
        # tensorboard stuff
        self.file_writer.add_summary(summary, t)

    def sample_path(self, env, num_episodes = None):
        """
        Sample paths (trajectories) from the environment.

        Args:
            num_episodes: the number of episodes to be sampled
                if none, sample one batch (size indicated by config file)
            env: open AI Gym envinronment

        Returns:
            paths: a list of paths. Each path in paths is a dictionary with
                path["observation"] a numpy array of ordered observations in the path
                path["actions"] a numpy array of the corresponding actions in the path
                path["reward"] a numpy array of the corresponding rewards in the path
            total_rewards: the sum of all rewards encountered during this "path"
        """
        episode = 0
        episode_rewards = []
        paths = []
        t = 0

        while (num_episodes or t < self.config.batch_size):
            state = env.reset()
            states, actions, rewards = [], [], []
            episode_reward = 0

            for step in range(self.config.max_ep_len):
                states.append(state)
                action = self.sess.run(self.sampled_actions, feed_dict={self.observation_placeholder : states[-1][None]})[0]
                state, reward, done, info = env.step(action)
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward
                t += 1
                if (done or step == self.config.max_ep_len-1):
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.config.batch_size:
                    break

            path = {"observation" : np.array(states),
                          "reward" : np.array(rewards),
                          "action" : np.array(actions)}
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards

    def close(self):
        self.sess.close()
        tf.reset_default_graph()

    def train(self):
        """
        Performs training
        """
        raise NotImplementedError()

    # def evaluate(self, env=None, num_episodes=1):
    #     """
    #     Evaluates the return for num_episodes episodes.
    #     Not used right now, all evaluation statistics are computed during training
    #     episodes.
    #     """
    #     if env==None: env = self.env
    #     paths, rewards = self.sample_path(env, num_episodes)
    #     avg_reward = np.mean(rewards)
    #     sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
    #     msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
    #     self.logger.info(msg)
    #     return avg_reward
    def save_model(self):
        save_path = os.path.join("./",self.config.output_path, self.agent_name, "run-{}".format(self.run))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join( save_path, "model.ckpt")
        save_path = self.saver.save(self.sess, save_path)
        print("Model saved in path: {}".format(save_path))

    def restore(self):
        load_path = os.path.join("./",self.config.output_path, self.agent_name, "run-{}".format(self.run))
        load_path = os.path.join(load_path, "model.ckpt")
        self.saver.restore(self.sess, load_path)

    def record(self):
         """
         Recreate an env and record a video for one episode
         """
         env = gym.make(self.config.env_name)
         record_path = os.path.join(self.config.record_path, self.agent_name, "run-{}".format(self.run))
         env = gym.wrappers.Monitor(env,record_path, video_callable=lambda x: True, resume=True)
         self.evaluate_policy(env, 1)