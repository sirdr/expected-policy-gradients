"""
Notes and Credits: 

The configuration classes within this file have been adapted from Homework 3
of the Winter 2019 version of Stanford's CS 234 taught by Emma Brunskill
"""


import tensorflow as tf

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
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches            = 100 # number of batches trained on
        self.batch_size             = 1000 # number of steps used to compute each policy update
        self.max_ep_len             = 1000 # maximum episode length
        self.learning_rate          = 10e-4
        self.critic_learning_rate   = 10e-3
        self.gamma                  = 0.99 # the discount factor
        self.normalize_advantage    = True 

        # parameters for the policy and baseline models
        self.n_layers               = 1
        self.layer_size             = 16
        self.activation             = tf.nn.relu 

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size

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
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches            = 100 # number of batches trained on
        self.batch_size             = 50000 # number of steps used to compute each policy update
        self.max_ep_len             = 1000 # maximum episode length
        self.learning_rate          = 10e-4
        self.critic_learning_rate   = 10e-3
        self.gamma                  = 0.99 # the discount factor
        self.normalize_advantage    = True
        self.target_update_weight   = 0.001

        # parameters for the policy and baseline models
        self.n_layers               = 3
        self.layer_size             = 32
        self.activation             = tf.nn.relu 

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size

def get_config(env_name):
    if env_name == 'pendulum':
        return config_pendulum()
    elif env_name == 'cheetah':
        return config_cheetah()
