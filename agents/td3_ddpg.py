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
from copy import copy

import tensorflow as tf

import gym
import scipy.signal
import os
import time
import inspect
from utils.general import get_logger, Progbar, export_plot
from config import get_config

from IPython import embed

from .base_pg import PG, Model

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', required=True, type=str,
                    choices=['cartpole','pendulum', 'cheetah'])


class Actor(Model):
    def __init__(self, num_actions, name='actor', max_action=1):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.max_action = max_action

    def __call__(self,
                obs,
                output_activation=None,
                reuse=False,
                training=True):

        output = None
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            #h = tf.layers.batch_normalization(obs, training=training)
            h = tf.layers.dense(obs, 400, activation=None, kernel_initializer=tf.initializers.variance_scaling())
            h = tf.nn.relu(h)
            h = tf.layers.dense(h, 300, activation=None, kernel_initializer=tf.initializers.variance_scaling())
            h = tf.nn.relu(h)
            out_weight_init = tf.initializers.random_uniform(-3e-3, 3e-3)
            h = tf.layers.dense(h, self.num_actions, activation=None, kernel_initializer=out_weight_init, bias_initializer=out_weight_init)
            output = self.max_action*tf.nn.tanh(h)
        return output


class Critic(Model):
    def __init__(self, name='critic', batch_norm=False):
        super().__init__(name=name)
        self.batch_norm = batch_norm

    def __call__(self, 
                obs, 
                actions,
                output_activation=None,
                reuse=False,
                training=True):

        output = None
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            #h = tf.layers.batch_normalization(obs, training=training)
            obs_act = tf.concat([obs, tf.cast(actions, tf.float32)], 1) 
            h = tf.layers.dense(obs_act, 400, activation=None, kernel_initializer=tf.initializers.variance_scaling())
            h = tf.nn.relu(h)
            h = tf.layers.dense(h, 300, activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling())
            out_weight_init = tf.initializers.random_uniform(-3e-3, 3e-3)
            output = tf.layers.dense(h, 1, activation=output_activation, kernel_initializer=out_weight_init, bias_initializer=out_weight_init)
        return output


class TD3DDPG(PG):
    """
    Class for Expected Policy Gradients, Inherets from the generic Policy Gradient class
    """
    def __init__(self, env, config, experience, actor=None, critic=None, action_noise = None, logger=None):
        super().__init__(env, config, logger=logger)
        if actor is None:
            actor = Actor(self.action_dim, max_action=self.action_high)
        self.actor = actor

        if critic is None:
            critic = Critic()
        self.critic = critic

        self.experience = experience

        target_actor = copy(actor)
        target_actor.name = 'target_actor'
        self.target_actor = target_actor

        target_critic = copy(critic)
        target_critic.name = 'target_critic'
        self.target_critic = target_critic

        # build model
        self.action_noise = action_noise
        self.tau = 0.005
        self.actor_lr = 0.001
        self.build()

    def add_placeholders_op(self):
        self.observation_placeholder = tf.placeholder(tf.float32, shape=(None, self.observation_dim))
        self.next_observation_placeholder = tf.placeholder(tf.float32, shape=(None, self.observation_dim))
        if self.discrete:
            self.action_placeholder = tf.placeholder(tf.int32, shape=(None, 1))
        else:
            self.action_placeholder = tf.placeholder(tf.float32, shape=(None, self.action_dim))

        self.reward_placeholder = tf.placeholder(tf.float32, shape=(None, 1))
        self.done_placeholder = tf.placeholder(tf.bool, shape=(None, 1))

        self.training_placeholder = tf.placeholder(tf.bool)

    def add_actor_loss_op(self):
        self.loss_integrand = tf.reduce_mean(self.critic_output_given_actor_output)
        self.loss = -1*self.loss_integrand

    def add_actor_optimizer_op(self):
        opt = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
        v_list = self.actor.vars
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # Needed for Batch Normalization to work properly
        self.actor_grads_and_vars = opt.compute_gradients(self.loss, var_list=v_list)
        self.train_op = opt.apply_gradients(self.actor_grads_and_vars)

    def add_critic_loss_op(self, scope="critic"):
        self.y = self.reward_placeholder + self.config.gamma*tf.multiply(tf.cast(tf.logical_not(self.done_placeholder), tf.float32), self.target_critic_output)
        self.critic_loss = tf.losses.mean_squared_error(self.y, self.critic_output,  scope=self.critic.name)

    def add_critic_optimizer_op(self):
        opt = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
        v_list = self.critic.vars
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.actor_grads_and_vars = opt.compute_gradients(self.critic_loss, var_list=v_list)
        self.train_critic_op = opt.apply_gradients(self.actor_grads_and_vars)

    def add_soft_update_target_op(self):
        critic_v_list = self.critic.vars
        target_critic_v_list = self.target_critic.vars
        self.update_target_critic_op = tf.group(*[tf.assign(target_v, self.tau*v + (1 - self.tau)*target_v) for v, target_v in zip(critic_v_list, target_critic_v_list)])

        actor_v_list = self.actor.vars
        target_actor_v_list = self.target_actor.vars
        self.update_target_actor_op = tf.group(*[tf.assign(target_v, self.tau*v + (1 - self.tau)*target_v) for v, target_v in zip(actor_v_list, target_actor_v_list)])

    def add_init_update_target_op(self):
        critic_v_list = self.critic.vars
        target_critic_v_list = self.target_critic.vars
        self.init_target_critic_op = tf.group(*[tf.assign(target_v, v) for v, target_v in zip(critic_v_list, target_critic_v_list)])

        actor_v_list = self.actor.vars
        target_actor_v_list = self.target_actor.vars
        self.init_target_actor_op = tf.group(*[tf.assign(target_v, v) for v, target_v in zip(actor_v_list, target_actor_v_list)])

    def build(self):
        # Observation normalization.
        # if self.normalize_observations:
        #     with tf.variable_scope('obs_rms'):
        #         self.obs_rms = RunningMeanStd(shape=self.observation_shape)
        # else:
        #     self.obs_rms = None
        # normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms),
        #     self.obs_low, self.obs_high)
        # normalized_obs1 = tf.clip_by_value(normalize(self.obs1, self.obs_rms),
        #     self.obs_low, self.obs_high)

        # add placeholders
        self.add_placeholders_op()
        # create policy net

        # Create networks and core TF parts that are shared across setup parts.
        # self.actor_tf = actor(normalized_obs0)
        # self.normalized_critic_tf = critic(normalized_obs0, self.actions)
        # self.critic_tf = denormalize(tf.clip_by_value(self.normalized_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        # self.normalized_critic_with_actor_tf = critic(normalized_obs0, self.actor_tf, reuse=True)
        # self.critic_with_actor_tf = denormalize(tf.clip_by_value(self.normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        # Q_obs1 = denormalize(target_critic(normalized_obs1, target_actor(normalized_obs1)), self.ret_rms)
        # self.target_Q = self.rewards + (1. - self.terminals1) * gamma * Q_obs1

        self.actor_output = self.actor(self.observation_placeholder, training=self.training_placeholder)
        self.target_actor_output = self.target_actor(self.next_observation_placeholder, training=self.training_placeholder)

        # get critic values
        self.critic_output = self.critic(self.observation_placeholder, self.action_placeholder, training=self.training_placeholder)
        self.target_critic_output = self.target_critic(self.next_observation_placeholder, self.target_actor_output, training=self.training_placeholder)

        self.critic_output_given_actor_output = self.critic(self.observation_placeholder, self.actor_output, training=self.training_placeholder)

        self.add_critic_loss_op()
        self.add_critic_optimizer_op()

        self.add_actor_loss_op()
        self.add_actor_optimizer_op()

        self.add_soft_update_target_op()
        self.add_init_update_target_op()

    def initialize(self):
        # create tf session
        self.sess = tf.Session()
        self.add_summary()
        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.sess.run([self.init_target_actor_op, self.init_target_critic_op])

    def update_critic(self, actions, observations, next_observations, rewards, done):
        self.sess.run(self.train_critic_op, feed_dict={
                        self.observation_placeholder : observations,
                        self.next_observation_placeholder : next_observations,
                        self.action_placeholder : actions,
                        self.reward_placeholder : rewards,
                        self.done_placeholder : done})

    def reset(self):
        self.action_noise.reset()

    def act(self, observation, apply_noise=True, compute_q=True):
        feed_dict = {self.observation_placeholder: observation[None]}
        if compute_q:
            action, q = self.sess.run([self.actor_output, self.critic_output_given_actor_output], feed_dict=feed_dict)
        else:
            action = self.sess.run(self.actor_output, feed_dict=feed_dict)
            q = None

        if self.action_noise is not None and apply_noise:
            noise = self.action_noise.sample()
            assert noise.shape == action[0].shape
            action += noise
        action = np.clip(action, self.action_low, self.action_high)
        return action, q

    def add_experience(self, observation, action, reward, new_observation, done):
        self.experience.add((observation, new_observation, action, reward,  done))

    def evaluate_policy(self, env=None, eval_episodes=10):
        if env==None: env = self.env
        rewards = []
        for _ in range(eval_episodes):
            obs = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _ = self.act(np.array(obs), apply_noise=False, compute_q=False)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
            rewards.append(total_reward)
        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        print("---------------------------------------")
        print("Evaluation over {} episodes: {:04.2f} +/- {:04.2f}".format(eval_episodes, avg_reward, sigma_reward))
        print("---------------------------------------")
        return avg_reward

    def train(self, iterations):

        for it in range(iterations):

            # Sample replay buffer 
            observations, next_observations, actions, rewards, dones = self.experience.sample(self.config.batch_size)

            self.update_critic(actions, observations, next_observations, rewards, dones)

            _, loss_integrand = self.sess.run([self.train_op, self.loss_integrand], feed_dict={
                        self.observation_placeholder : observations,
                        self.action_placeholder : actions})
            #print("loss integrand: {}".format(loss_integrand))

            self.sess.run([self.update_target_critic_op, self.update_target_actor_op], feed_dict={})

        

if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config(args.env_name)
    env = gym.make(config.env_name)
    # train model
    model = EPG(env, config)

    model.run()