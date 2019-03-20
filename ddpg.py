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
from experience import ReplayBuffer
from noise import NormalActionNoise
from agents.td3_ddpg import TD3DDPG

import pickle

from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', required=True, type=str,
                    choices=['cartpole','pendulum', 'cheetah'])
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--save_as', type=str, default='results')
parser.add_argument('--num_episodes', type=int, default=2000)
parser.add_argument('--num_eval_final', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--seed', type=int, default=7)

# def evaluate_policy(agent, eval_episodes=10):
#     avg_reward = 0.
#     for _ in range(eval_episodes):
#         obs = env.reset()
#         done = False
#         while not done:
#             action, _ = agent.act(np.array(obs), apply_noise=False, compute_q=False)
#             obs, reward, done, _ = env.step(action)
#             avg_reward += reward

#     avg_reward /= eval_episodes

#     print("---------------------------------------")
#     print("Evaluation over {} episodes: {}".format(eval_episodes, avg_reward))
#     print("---------------------------------------")
#     return avg_reward

def learn(env, config, num_episodes = 5000, num_eval_final = 50, batch_size = 100, seed = 7, run=0):
    """
    Apply procedures of training for a DDPG.
    """

    experience = ReplayBuffer()
    noise = NormalActionNoise(0, 0.1, size=env.action_space.shape[0])

    config.batch_size = batch_size

    # initialize
    agent = TD3DDPG(env, config, experience, action_noise = noise, run=run)
    agent.initialize()

    # record one game at the beginning
    # if agent.config.record:
    #     agent.record()
    # model
        
    # Evaluate untrained policy
    agent.evaluate_policy()

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True 

    stats = {}
    stats["episode_rewards"] = []
    stats["evaluation_rewards"] = []
    stats["grad_norms"] = [] # TODO
    stats["cummulative_timesteps"] = []
    stats["episode_timesteps"] = []
    stats["eval_episode_timesteps"] = []
    stats["eval_cummulative_timesteps"] = []
    stats["num_episodes"] = num_episodes
    stats["num_eval_final"] = num_eval_final
    stats["seed"] = seed
    stats["agent"] = agent.agent_name
    stats["env"] = config.env_name

    while episode_num < num_episodes: #total_timesteps < config.max_timesteps:
        
        if done: 

            if total_timesteps != 0: 
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(total_timesteps, episode_num, episode_timesteps, episode_reward))
                stats = agent.train(episode_timesteps, stats=stats)
                stats["episode_rewards"].append(episode_reward)
                stats["cummulative_timesteps"].append(total_timesteps)
                stats["episode_timesteps"].append(episode_timesteps)
            
            # Evaluate episode
            # if timesteps_since_eval >= config.eval_freq:
            #     timesteps_since_eval %= config.eval_freq
            #     evaluations.append(agent.evaluate_policy())
                
            
            # Reset environment
            observation = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
        
        # Select action randomly or according to policy
        if total_timesteps < config.start_timesteps:
            action = env.action_space.sample()
        else:
            action, _ = agent.act(np.array(observation), apply_noise=True, compute_q=False)

        # Perform action
        new_observation, reward, done, _ = env.step(action) 
        done_bool = False if episode_timesteps + 1 == env._max_episode_steps else done
        episode_reward += reward

        # Store data in replay buffer
        agent.add_experience(observation, action, reward, new_observation, done_bool)

        observation = new_observation

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        
    # Final evaluation 
    rewards, eval_cummulative_timesteps, eval_episode_timesteps = agent.evaluate_policy(eval_episodes=num_eval_final)
    stats["evaluation_rewards"] = rewards
    stats["eval_cummulative_timesteps"] = eval_cummulative_timesteps
    stats["eval_episode_timesteps"] = eval_episode_timesteps

    agent.save_model()
    agent.record()
    agent.close()

    return stats


    # if agent.config.record:
    #     agent.record()



if __name__ == '__main__':

    args = parser.parse_args()
    config = get_config(args.env_name)
    env = gym.make(config.env_name)
    outfile = "{}.pickle".format(args.save_as)
    outpath = os.path.join(config.output_path, "td3ddpg",  outfile)
    runs = {}
    seed = args.seed
    env.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    for i in range(args.runs):
        # train model
        stats_dict = learn(env, config, 
                    num_episodes=args.num_episodes, 
                    num_eval_final=args.num_eval_final, 
                    batch_size=args.batch_size,
                    seed=seed,
                    run=i)
        runs[i] = stats_dict
    # save dictionary 
    pickle_out = open(outpath,"wb")
    pickle.dump(runs, pickle_out)
    pickle_out.close()
    pickle_in = open(outpath,"rb")
    example_dict = pickle.load(pickle_in)
    print(example_dict)
