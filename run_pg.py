# -*- coding: UTF-8 -*-

"""
Expected Policy Gradients
Author: Loren Amdahl-Culleton

Notes and Credits: 

Much of the structure and some of the functions in run_pg.py have been adapted
from Homework 3 of the Winter 2019 version of Stanford's CS 234 taught by Emma Brunskill

"""


import os
import argparse
import sys
import logging
import time
import numpy as np
from scipy import integrate

import tensorflow as tf

import gym
import scipy.signal
import os
import time
import inspect
from utils.general import get_logger, Progbar, export_plot
from config import get_config

from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', required=True, type=str,
                    choices=['pendulum', 'cheetah'])

def build_mlp(
          mlp_input,
          output_size,
          scope,
          actions_input = None,
          output_activation=None):
    """

    """
    output = None
    with tf.variable_scope(scope):
        h = tf.layers.dense(mlp_input, 400, activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling())
        if actions_input is not None:
            h = tf.concatenate([h, actions_input]) # this is for the Q network
        h = tf.layers.dense(h, 300, activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling())
    out_weight_init = tf.initializers.random_uniform(-3e-3, 3e-3)
    output = tf.layers.dense(h, output_size, activation=output_activation, kernel_initializer=out_weight_init, bias_initializer=out_weight_init)
    return output


class PG(object):
    """
    Abstract Class for implementing a Policy Gradient Based Algorithm
    """
    def __init__(self, env, config, logger=None):
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
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]

        self.lr = self.config.learning_rate

        # build model
        self.build()

    def add_placeholders_op(self):
        """
        Add necessary placeholders for policy gradient algorithm
        """
        raise NotImplementedError()

    def build_policy_network_op(self, scope = "policy_network"):
        """
        Build the policy network.
        """
        raise NotImplementedError()


    def add_loss_op(self):
        """
        Compute the loss, averaged for a given batch.

        """
        raise NotImplementedError()

    def add_optimizer_op(self):
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
        """
        Assumes the graph has been constructed (have called self.build())
        Creates a tf Session and run initializer of variables

        You don't have to change or use anything here.
        """
        # create tf session
        self.sess = tf.Session()
        # tensorboard stuff
        self.add_summary()
        # initiliaze all variables
        init = tf.global_variables_initializer()
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
        self.file_writer = tf.summary.FileWriter(self.config.output_path,self.sess.graph)

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
                action = self.sess.run(self.sampled_action, feed_dict={self.observation_placeholder : states[-1][None]})[0]
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

    def train(self):
        """
        Performs training
        """
        raise NotImplementedError()

    def evaluate(self, env=None, num_episodes=1):
        """
        Evaluates the return for num_episodes episodes.
        Not used right now, all evaluation statistics are computed during training
        episodes.
        """
        if env==None: env = self.env
        paths, rewards = self.sample_path(env, num_episodes)
        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        self.logger.info(msg)
        return avg_reward

    def record(self):
         """
         Recreate an env and record a video for one episode
         """
         env = gym.make(self.config.env_name)
         env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
         self.evaluate(env, 1)

    def run(self):
        """
        Apply procedures of training for a PG.
        """
        # initialize
        self.initialize()
        # record one game at the beginning
        if self.config.record:
            self.record()
        # model
        self.train()
        # record one game at the end
        if self.config.record:
            self.record()


class EPG(PG):
    """
    Class for Expected Policy Gradients, Inherets from the generic Policy Gradient class
    """
    def __init__(self, env, config, logger=None):
        """
        Initialize Policy Gradient Class

        Args:
                env: an OpenAI Gym environment
                config: class with hyperparameters
                logger: logger instance from the logging module

        """
        # directory for training outputs
        super().__init__(env, config, logger=logger)

        self.lr_critic = self.config.critic_learning_rate
        self.upper_bound = self.env.action_space.high
        self.lower_bound = self.env.action_space.low

    def add_placeholders_op(self):
        """
        Add placeholders for observation, action, and advantage:
            self.observation_placeholder, type: tf.float32
            self.action_placeholder, type: depends on the self.discrete
            self.advantage_placeholder, type: tf.float32
        """
        self.observation_placeholder = tf.placeholder(tf.float32, shape=(None, self.observation_dim))
        if self.discrete:
            self.action_placeholder = tf.placeholder(tf.int32, shape=(None,))
        else:
            self.action_placeholder = tf.placeholder(tf.float32, shape=(None, self.action_dim))

        self.reward_placeholder = tf.placeholder(tf.float32, shape=(None))

    def build_policy_network_op(self, scope = "policy_network"):
        """

        """
        if self.discrete:
            action_logits = build_mlp(self.observation_placeholder, self.action_dim , scope)
            self.sampled_action = tf.squeeze(tf.multinomial(action_logits, 1), axis = 1)
            self.logprob = -1*tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.action_placeholder, logits=action_logits)
            self.prob = np.exp(logprob)
        else:
            action_means = build_mlp(self.observation_placeholder, self.action_dim , scope)
            with tf.variable_scope(scope):
                log_std = tf.get_variable("log_std", shape=(self.action_dim))
            shape = tf.shape(action_means)
            epsilon = tf.random_normal(shape)
            self.sampled_action = action_means + tf.multiply(epsilon, tf.exp(log_std))
            self.logprob = tf.contrib.distributions.MultivariateNormalDiag(loc=action_means, scale_diag=tf.exp(log_std)).log_prob(self.action_placeholder)
            self.prob = np.exp(logprob)

    def add_critic_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable

        """
        TODO: 
            The loss for an example is defined as:
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2 
        HINT: 
            - Config variables are accessible through self.config
            - You can access placeholders like self.a (for actions)
                self.r (rewards) or self.done_mask for instance
            - You may find the following functions useful
                - tf.cast
                - tf.reduce_max
                - tf.reduce_sum
                - tf.one_hot
                - tf.squared_difference
                - tf.reduce_mean
        """
        num_actions = self.env.action_space.n

        q_samp = self.r + self.config.gamma*tf.multiply(tf.cast(tf.logical_not(self.done_mask), tf.float32),(tf.reduce_max(target_q, axis=1)))
        action_mask = tf.one_hot(self.a, num_actions)
        self.loss = tf.reduce_mean(tf.squared_difference(q_samp,tf.reduce_sum(tf.multiply(action_mask, q), axis=1)))


    def add_gradients_op(self):
        """
        Compute the gradients using the EPG formulation.

        This includes calculating the integral over the action space.

        """
        ranges = [(self.lower_bound, self.upper_bound)]*self.action_dim
        def func(x0):
            input = np.array([x0,])
            assert state.shape[0] == x0.shape[0]

            qvals = self.sess.run(self.critic, feed_dict={self.observation_placeholder : state, self.action_placeholder : input} )


            return val

        grad = self.config.gamma*integrate.nquad(func, ranges)

        lambda y: tf.py_func(scipy.integrate.quad( f, self.lower_bound, self.upper_bound)[ 0 ], [self.sampled_action], tf.float64)
        self.grads = -1*tf.reduce_mean(tf.multiply(self.logprob, self.advantage_placeholder))

    def add_optimizer_op(self):
        """
        """
        opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = opt.apply_gradients(self.grads)

    def add_critic_op(self, scope = "critic"):
        """
        Builds critic network, target critic network, and adds loss

        """
        critic_output = build_mlp(self.observation_placeholder, 1 , scope, actions_input=self.action_placeholder)
        self.critic = tf.squeeze(critic_output, axis = 1)
        critic_target_output = build_mlp(self.observation_placeholder, 1 , "target_critic", actions_input=self.action_placeholder)
        self.critic_target = tf.squeeze(critic_target_output, axis = 1)
        loss = tf.losses.mean_squared_error(self.critic_target, self.critic, scope=scope)
        opt = tf.train.AdamOptimizer(learning_rate=self.lr_critic)
        self.update_critic_op = opt.minimize(loss)

    def build(self):
        """
        Build the model by adding all necessary variables.

        You don't have to change anything here - we are just calling
        all the operations you already defined above to build the tensorflow graph.
        """

        # add placeholders
        self.add_placeholders_op()
        # create policy net
        self.build_policy_network_op()
        # create critic net
        self.build_critic_network_op()
        #create target critic
        self.build_critic_network_op(scope="target_critic_network")
        # add square loss
        self.add_loss_op()
        # add optmizer for the main networks
        self.add_optimizer_op()

        # add baseline
        if self.config.use_baseline:
            self.add_baseline_op()

    def update_critic(self, actions, observations):
        """
        Update the baseline from given returns and observation.

        Args:
                actions: action sampled from policy
                observations: observations
        """

        self.sess.run(self.update_critic_op, feed_dict={
                        self.observation_placeholder : observations,
                        self.action_placeholder : actions})

    def train(self):
        """
        Performs training

        You do not have to change or use anything here, but take a look
        to see how all the code you've written fits together!
        """
        last_eval = 0
        last_record = 0

        self.init_averages()
        scores_eval = [] # list of scores computed at iteration time

        episode = 0
        episode_rewards = []
        paths = []

        observation = self.env.reset

        for t in range(self.config.num_batches*self.config.batch_size):

            self.sess.run(self.train_op, feed_dict={
                        self.observation_placeholder : observation[None]})

            action = self.sess.run(self.sampled_action, feed_dict={self.observation_placeholder : observation[None]})[0]
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1

            self.sess.run()

            if (done or step == self.config.max_ep_len-1):
                episode_rewards.append(episode_reward)
                observation = self.env.reset
                episode_reward = 0
                episode += 1
                step = 0

            # tf stuff
            if (t % (self.config.summary_freq*self.config.batch_size) == 0):
                self.update_averages(episode_rewards, scores_eval)
                self.record_summary(t)

            if (t % self.config.batch_size == 0):
                # compute reward statistics for this batch and log
                avg_reward = np.mean(episode_rewards)
                sigma_reward = np.sqrt(np.var(episode_rewards) / len(episode_rewards))
                msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
                self.logger.info(msg)
                scores_eval = scores_eval + episode_rewards
                episode_rewards = []

            if  self.config.record and (last_record > (self.config.record_freq*self.config.batch_size)):
                self.logger.info("Recording...")
                last_record =0
                self.record()

        self.logger.info("- Training done.")
        export_plot(scores_eval, "Score", config.env_name, self.config.plot_output)



if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config(args.env_name)
    env = gym.make(config.env_name)
    embed()
    exit()
    # train model
    model = EPG(env, config)

    model.run()