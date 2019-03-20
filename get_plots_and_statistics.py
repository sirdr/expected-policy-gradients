
import os
import argparse
import sys
import logging
import time
import numpy as np
from scipy import integrate

import matplotlib.pyplot as plt

import tensorflow as tf

import gym
import scipy.signal
import os
import time
import inspect
from utils.general import get_logger, Progbar, export_plot
from config import get_config
from experience import ReplayBuffer
from noise import NormalActionNoise
from agents.td3_ddpg import TD3DDPG

import pickle

from IPython import embed


parser.add_argument('--env_name', required=True, type=str,
                    choices=['cartpole','pendulum', 'cheetah'])

path = "./results/{}"

def get_plots(env, config):


if __name__ == '__main__':

    args = parser.parse_args()
    config = get_config(args.env_name)
    env = gym.make(config.env_name)

    path = os.path.join("./",self.config.output_path)

    path

    main()