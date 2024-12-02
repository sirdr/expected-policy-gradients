# -*- coding: UTF-8 -*-

"""
Author: Loren Amdahl-Culleton

Notes and Credits: 

evaluate_policy method has been adapted from Homework 3 
of the Winter 2019 version of Stanford's CS 234 taught
 by Emma Brunskill

"""


import argparse
import numpy as np
from copy import copy

import tensorflow as tf

import gym
from config import get_config

from .base_pg import PG

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', required=True, type=str,
                    choices=['cartpole','pendulum', 'cheetah'])


class Actor(tf.keras.Model):
    def __init__(self, num_actions, name='actor', max_action=1):
        super(Actor, self).__init__(name=name)
        self.num_actions = num_actions
        self.max_action = max_action
        self.layer1 = tf.keras.layers.Dense(400, activation='relu', kernel_initializer=tf.compat.v1.initializers.variance_scaling())
        self.layer2 = tf.keras.layers.Dense(300, activation='relu', kernel_initializer=tf.compat.v1.initializers.variance_scaling())
        self.layer3 = tf.keras.layers.Dense(
            self.num_actions, 
            activation = 'tanh',
            kernel_initializer=tf.compat.v1.initializers.random_uniform(-3e-3, 3e-3), 
            bias_initializer=tf.compat.v1.initializers.random_uniform(-3e-3, 3e-3)
        )

    def __call__(self, obs, training=False):
        h = self.layer1(obs)
        h = self.layer2(h)
        h = self.layer3(h)
        return self.max_action * h
    
    def get_config(self):
        return {
            "num_actions": self.num_actions,
            "max_action": self.max_action,
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Critic(tf.keras.Model):
    def __init__(self, name='critic', batch_norm=False):
        super(Critic, self).__init__(name=name)
        self.batch_norm = batch_norm
        self.layer1 = tf.keras.layers.Dense(400, activation='relu', kernel_initializer=tf.compat.v1.initializers.variance_scaling())
        self.layer2 = tf.keras.layers.Dense(300, activation='relu', kernel_initializer=tf.compat.v1.initializers.variance_scaling())
        self.layer3 = tf.keras.layers.Dense(
            1, 
            activation=None,
            kernel_initializer=tf.compat.v1.initializers.random_uniform(-3e-3, 3e-3), 
            bias_initializer=tf.compat.v1.initializers.random_uniform(-3e-3, 3e-3)
        )

    def __call__(self, obs, actions, training=False):
        # Concatenate observations and actions along the last axis
        obs_act = tf.concat([obs, tf.cast(actions, tf.float32)], axis=-1)
        h = self.layer1(obs_act)
        h = self.layer2(h)
        return self.layer3(h)
    
    def get_config(self):
        return {
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TD3DDPG(PG):
    """Twin Delayed DDPG (TD3). 
    Ref. Fujimoto, Scott et al. “Addressing Function Approximation Error 
    in Actor-Critic Methods.” International Conference on Machine Learning (2018).

    See https://arxiv.org/pdf/1802.09477 for additional details.

    """
    def __init__(
        self, 
        env, 
        config,
        run_dir, 
        experience, 
        actor=None, 
        critic=None, 
        action_noise = None, 
        logger=None, 
        run=0
        ) -> None:
        super().__init__(env, config, run_dir, run=run, logger=logger)
        self.agent_name = "td3ddpg" # for model saving and restoring
        self.experience = experience
        self.action_noise = action_noise
        self.tau = 0.005 # overrides config
        self.actor_lr = 0.001 # overrides config

        # Initialize actor and critic
        self.actor = actor or Actor(self.action_dim, max_action=self.action_high)
        self.critic = critic or Critic()

        # Initialize target actor and critic
        target_actor = copy(self.actor)
        target_actor.name = 'target_actor'
        self.target_actor = target_actor
        target_critic = copy(self.critic)
        target_critic.name = 'target_critic'
        self.target_critic = target_critic

        # Initialize optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

    def initialize(self):
        """
        Initialize the agent including variables and checkpoint setup.
        """
        # Add summaries (TensorBoard writer setup)
        self.add_summary()

        # Initialize checkpoint system
        self.checkpoint = tf.train.Checkpoint(
            actor=self.actor,
            critic=self.critic,
            target_actor=self.target_actor,
            target_critic=self.target_critic,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.config.checkpoint_dir, max_to_keep=5
        )

        # Restore from the latest checkpoint if available
        if self.checkpoint_manager.latest_checkpoint:
            print(f"Restoring from checkpoint: {self.checkpoint_manager.latest_checkpoint}")
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        else:
            print("No checkpoint found. Initializing target networks.")

        # Initialize target networks
        for base_var, target_var in zip(self.actor.variables, self.target_actor.variables):
            target_var.assign(base_var)

        for base_var, target_var in zip(self.critic.variables, self.target_critic.variables):
            target_var.assign(base_var)

        print("Initialization complete.")


    @tf.function
    def update_critic(self, observations, actions, next_observations, rewards, done):
        # Remove extra dimensions if present
        if len(next_observations.shape) == 3:
            next_observations = tf.squeeze(next_observations, axis=1)
        if len(actions.shape) == 3:
            actions = tf.squeeze(actions, axis=1)
            # Cast rewards to float32
        rewards = tf.cast(rewards, tf.float32)
        next_actions = self.target_actor(next_observations, training=False)
        target_critic_output = self.target_critic(next_observations, next_actions, training=False)
        y = rewards + self.config.gamma*tf.multiply(tf.cast(tf.logical_not(done), tf.float32), target_critic_output)

        with tf.GradientTape() as tape:
            critic_output = self.critic(observations, actions, training=True)
            critic_loss = tf.reduce_mean(tf.square(y - critic_output))
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
    
    @tf.function
    def update_target(self, base, target, tau):
        for base_vars, target_vars in zip(base.variables, target.variables):
            base_vars.assign(tau * base_vars + (1 - tau) * target_vars)

    def reset(self):
        self.action_noise.reset()

    def act(self, observation, apply_noise=True, compute_q=True):

        action = self.actor(observation)

        if compute_q:
            q = self.critic(observation, action, training=False)
        else:
            q = None

        if self.action_noise is not None and apply_noise:
            noise = self.action_noise.sample()
            assert noise.shape == action[0].shape
            action += noise
        action = np.clip(action, self.action_low, self.action_high)

        # Remove batch dimension for single action environments
        action = np.squeeze(action, axis=0)

        return action, q

    def add_experience(self, observation, action, reward, new_observation, done):
        self.experience.add((observation, new_observation, action, reward,  done))

    def evaluate_policy(self, env=None, eval_episodes=10):
        if env==None: env = self.env
        rewards = []
        eval_cummulative_timesteps = []
        eval_episode_timesteps = []

        total_timesteps = 0
        episode_timesteps = 0
        for _ in range(eval_episodes):
            obs = env.reset()
            done = False
            total_reward = 0.0
            while not done:

                # Extract the actual observation if it's a tuple
                if isinstance(obs, tuple):
                    obs = obs[0]

                # Ensure observation is a NumPy array
                obs = np.array(obs, dtype=np.float32)

                # Add a batch dimension if the observation is 1D
                if obs.ndim == 1:
                    obs = obs[None, :]  # Add batch dimension

                action, _ = self.act(np.array(obs), apply_noise=False, compute_q=False)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                total_reward += reward
                episode_timesteps += 1
                total_timesteps += 1
            eval_episode_timesteps.append(episode_timesteps)
            eval_cummulative_timesteps.append(total_timesteps)
            rewards.append(total_reward)
            episode_timesteps = 0

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        print("---------------------------------------")
        print("Evaluation over {} episodes: {:04.2f} +/- {:04.2f}".format(eval_episodes, avg_reward, sigma_reward))
        print("---------------------------------------")
        return rewards, eval_cummulative_timesteps, eval_episode_timesteps

    def train(self, iterations, stats=None):
        if stats is None:
            stats = {"grad_norms": []}

        for _ in range(iterations):
            
            # Sample replay buffer 
            observations, next_observations, actions, rewards, dones = self.experience.sample(self.config.batch_size)

            # Update critic
            self.update_critic(actions, observations, next_observations, rewards, dones)

            # Update actor
            with tf.GradientTape() as tape:
                actions = self.actor(observations, training=True)
                critic_output = self.critic(observations, actions, training=False)
                actor_loss = -tf.reduce_mean(critic_output)

            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            actor_grad_norm = tf.linalg.global_norm(actor_grads)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            stats["grad_norms"].append(actor_grad_norm)

            # Update target networks
            self.update_target(self.actor, self.target_actor, self.tau)
            self.update_target(self.critic, self.target_critic, self.tau)

        return stats

        

if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config(args.env_name)
    env = gym.make(config.env_name)
    # train model
    model = TD3DDPG(env, config)

    model.run()