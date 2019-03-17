"""
Notes and Credits: 

The configuration classes within this file have been adapted from Homework 3
of the Winter 2019 version of Stanford's CS 234 taught by Emma Brunskill
"""


import tensorflow as tf

class config_cartpole:
    def __init__(self):
        self.env_name="CartPole-v0"
        self.record = True 

        # output config
        self.output_path = "results/{}/".format(self.env_name)
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path 
        self.eval_freq = 5e3
        self.summary_freq = 1

        # model and training config
        self.num_batches            = 100 # number of batches trained on
        self.batch_size             = 64 # number of steps used to compute each policy update
        self.max_ep_len             = 1000 # maximum episode length
        self.max_timesteps          = 5e4
        self.start_timesteps        = 1e4
        self.max_ep_len             = 1000 # maximum episode length
        self.learning_rate          = 1e-4
        self.critic_learning_rate   = 1e-3
        self.gamma                  = 0.99 # the discount factor
        self.target_update_weight   = 0.001


class config_pendulum:
    def __init__(self):
        self.env_name="InvertedPendulum-v1"
        self.record = True 

        # output config
        self.output_path = "results/{}/".format(self.env_name)
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path 
        self.eval_freq = 5e3
        self.summary_freq = 1

        # model and training config
        self.num_batches            = 1000 # number of batches trained on
        self.batch_size             = 64 # number of steps used to compute each policy update
        self.max_ep_len             = 1000 # maximum episode length
        self.max_timesteps          = 5e4
        self.start_timesteps        = 1e4
        self.max_ep_len             = 1000 # maximum episode length
        self.learning_rate          = 1e-4
        self.critic_learning_rate   = 1e-3
        self.gamma                  = 0.99 # the discount factor
        self.target_update_weight   = 0.001


class config_cheetah:
    def __init__(self):
        self.env_name="HalfCheetah-v1"
        self.record = True 

        # output config
        self.output_path = "results/{}/".format(self.env_name)
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path 
        self.eval_freq = 5e3
        self.summary_freq = 1

        # model and training config
        self.max_timesteps          = 1e6
        self.start_timesteps        = 1e4
        self.batch_size             = 64 # number of steps used to compute each policy update
        self.max_ep_len             = 1000 # maximum episode length
        self.learning_rate          = 1e-4
        self.critic_learning_rate   = 1e-3
        self.gamma                  = 0.99 # the discount factor
        self.target_update_weight   = 0.001


def get_config(env_name):
    if env_name == 'pendulum':
        return config_pendulum()
    elif env_name == 'cartpole':
        return config_cartpole()
    elif env_name == 'cheetah':
        return config_cheetah()
