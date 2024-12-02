"""
Notes and Credits: 

Much of the structure and some of the functions in this file have been adapted
from Homework 3 of the Winter 2019 version of Stanford's CS 234 taught by Emma Brunskill

"""


import os
from datetime import datetime
import numpy as np

import tensorflow as tf

import gym
import os
from utils.general import get_logger


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)



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
        """
        Initialize the agent, including TensorFlow 2.x-specific setups like checkpoints and summary writers.
        """
        # Initialize TensorBoard summary writer and variables
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
            print("No checkpoint found. Initializing from scratch.")

        # Ensure variables are initialized
        for variable in self.actor.variables + self.critic.variables + self.target_actor.variables + self.target_critic.variables:
            variable.assign(variable)  # Triggers lazy initialization if needed


    def add_summary(self):
        """
        Tensorboard setup for TensorFlow 2.x.
        """
        # Create a summary writer for TensorBoard logging
        self.writer = tf.summary.create_file_writer(self.config.output_path)

        # Create variables to store reward statistics
        self.avg_reward = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="avg_reward")
        self.max_reward = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="max_reward")
        self.std_reward = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="std_reward")
        self.eval_reward = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="eval_reward")
    
    def log_summary(self, step):
        """
        Log scalar metrics to TensorBoard.
        """
        with self.writer.as_default():
            tf.summary.scalar("Avg Reward", self.avg_reward, step=step)
            tf.summary.scalar("Max Reward", self.max_reward, step=step)
            tf.summary.scalar("Std Reward", self.std_reward, step=step)
            tf.summary.scalar("Eval Reward", self.eval_reward, step=step)
            self.writer.flush()


    def init_averages(self):
        """
        Initialize reward statistics.
        """
        self.avg_reward.assign(0.0)
        self.max_reward.assign(0.0)
        self.std_reward.assign(0.0)
        self.eval_reward.assign(0.0)


    def update_averages(self, rewards, scores_eval):
        """
        Update the averages.

        Args:
            rewards: deque
            scores_eval: list
        """
        self.avg_reward.assign(np.mean(rewards))
        self.max_reward.assign(np.max(rewards))
        self.std_reward.assign(np.sqrt(np.var(rewards) / len(rewards)))

        if len(scores_eval) > 0:
            self.eval_reward.assign(scores_eval[-1])


    def record_summary(self, t):
        """
        Add summary to TensorBoard.
        """
        with self.writer.as_default():
            tf.summary.scalar("Avg Reward", self.avg_reward, step=t)
            tf.summary.scalar("Max Reward", self.max_reward, step=t)
            tf.summary.scalar("Std Reward", self.std_reward, step=t)
            tf.summary.scalar("Eval Reward", self.eval_reward, step=t)
            self.writer.flush()


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
        """
        Perform cleanup tasks such as closing TensorBoard writers.
        """
        if hasattr(self, 'writer') and self.writer:  # Check if TensorBoard writer exists
            self.writer.close()
            print("TensorBoard writer closed.")
        print("Cleanup complete.")

    def train(self):
        """
        Performs training
        """
        raise NotImplementedError()

    def evaluate(self, env=None, num_episodes=1):
        """
        Evaluates the return for num_episodes episodes.
        """
        if env==None: env = self.env
        paths, rewards = self.sample_path(env, num_episodes)
        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        self.logger.info(msg)
        return avg_reward

    def save_model(self):
        """
        Save the model using the CheckpointManager.
        """
        if self.checkpoint_manager:
            save_path = self.checkpoint_manager.save()
            print(f"Model saved to {save_path}")
        else:
            print("CheckpointManager is not initialized. Cannot save the model.")


    def restore(self):
        """
        Restore the model using the latest checkpoint.
        """
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f"Model restored from {self.checkpoint_manager.latest_checkpoint}")
        else:
            print("No checkpoint found to restore from.")


    def record(self):
        """
        Recreate an env and record a video for one episode
        """
        env = gym.make(self.config.env_name, render_mode="rgb_array")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        record_path = os.path.join(self.config.record_path, self.agent_name, "run-{}".format(self.run), timestamp)
        if not os.path.exists(record_path):
            os.makedirs(record_path)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=record_path,
            episode_trigger=lambda x: True  # Record every episode
        )
        # Evaluate the policy in the wrapped environment
        self.evaluate_policy(env, eval_episodes=1)
        # Close the environment after recording
        env.close()