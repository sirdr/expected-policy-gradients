# -*- coding: UTF-8 -*-

"""
Author: Loren Amdahl-Culleton

Notes and Credits: 

Some of the structure in this file has been adapted from Homework 3 of the 
Winter 2019 version of Stanford's CS 234 taught by Emma Brunskill
"""


import os
import argparse
import numpy as np

from datetime import datetime

import tensorflow as tf

import gym
import os
from config import get_config
from experience import ReplayBuffer
from noise import NormalActionNoise
from agents.td3_ddpg import TD3DDPG

from utils.general import create_run_directory, save_config

import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', required=True, type=str,
                    choices=['cartpole','pendulum', 'cheetah'])
parser.add_argument('--output_dir', required=True, help="Base directory for storing run outputs")
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--save_as', type=str, default='results')
parser.add_argument('--num_episodes', type=int, default=2000)
parser.add_argument('--num_eval_final', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--record', action='store_true')
parser.add_argument('--eval_from_checkpoint', action="store_true")

def learn(env, config, run_dir, num_episodes = 5000, num_eval_final = 50, batch_size = 100, seed = 7, run=0, record=False):
    """
    Apply procedures of training for a DDPG.
    """

    experience = ReplayBuffer()
    noise = NormalActionNoise(0, 0.1, size=env.action_space.shape[0])

    config.batch_size = batch_size

    # initialize
    agent = TD3DDPG(env, config, run_dir, experience, action_noise = noise, run=run)
    agent.initialize()
        
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

    while episode_num < num_episodes:
        
        if done: 

            if total_timesteps != 0: 
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(total_timesteps, episode_num, episode_timesteps, episode_reward))
                stats = agent.train(episode_timesteps, stats=stats)
                stats["episode_rewards"].append(episode_reward)
                stats["cummulative_timesteps"].append(total_timesteps)
                stats["episode_timesteps"].append(episode_timesteps)
            
            # Reset environment
            observation = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
            
            # Extract the actual observation if it's a tuple
            if isinstance(observation, tuple):
                observation = observation[0]

            # Ensure observation is a NumPy array
            observation = np.array(observation, dtype=np.float32)

            # Add a batch dimension if the observation is 1D
            if observation.ndim == 1:
                observation = observation[None, :]  # Add batch dimension
        
        # Select action randomly or according to policy
        if total_timesteps < config.start_timesteps:
            action = env.action_space.sample()
        else:
            action, _ = agent.act(np.array(observation), apply_noise=True, compute_q=False)
            # Only squeeze if action has a singleton batch dimension
            if action.ndim > 1 and action.shape[0] == 1:
                action = np.squeeze(action, axis=0)

        # Perform action
        new_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        done_bool = False if episode_timesteps + 1 == env._max_episode_steps else done
        episode_reward += reward

        # Extract the actual observation if it's a tuple
        if isinstance(new_observation, tuple):
            new_observation = new_observation[0]

        # Ensure observation is a NumPy array
        new_observation = np.array(new_observation, dtype=np.float32)

        # Add a batch dimension if the observation is 1D
        if new_observation.ndim == 1:
            new_observation = new_observation[None, :]  # Add batch dimension

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
    if record:
        agent.record()
    agent.close()

    return stats


def eval_from_checkpoint(env, config, run=0):
    """
    Apply procedures of training for a DDPG.
    """

    experience = ReplayBuffer()
    noise = NormalActionNoise(0, 0.1, size=env.action_space.shape[0])

    # initialize
    agent = TD3DDPG(env, config, experience, action_noise = noise, run=run)
    agent.initialize()

    agent.restore()
    agent.record()
    agent.close()




if __name__ == '__main__':

    args = parser.parse_args()
    config = get_config(args.env_name)
    seed = args.seed
    env = gym.make(config.env_name)
    env.reset(seed=seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    agent_dir = os.path.join(args.output_dir, 'td3')

    if args.eval_from_checkpoint:
        for i in range(args.runs):
            # train model
            eval_from_checkpoint(env, config, run=i)
    else:
        outpaths = {}
        for i in range(args.runs):
            # train model

            # Create a unique directory for this run
            run_dir = create_run_directory(agent_dir)
            save_config(vars(args), run_dir)
            stats_dict = learn(env, config, run_dir, 
                        num_episodes=args.num_episodes, 
                        num_eval_final=args.num_eval_final, 
                        batch_size=args.batch_size,
                        seed=seed,
                        run=i,
                        record = args.record)
            outdir = os.path.join(run_dir, "results")
            # save dictionary 
            outpath = os.path.join(outdir, f"results.pickle")
            pickle_out = open(outpath,"wb")
            pickle.dump(stats_dict, pickle_out)
            pickle_out.close()
            outpaths[i] = outpath
        
        for i, outpath in outpaths.items():
            pickle_in = open(outpath,"rb")
            stats_dict = pickle.load(pickle_in)
            print("Stored the following runs:")
            print(f"run {i} | number of keys in stats dict: {len(stats_dict.keys())}")

