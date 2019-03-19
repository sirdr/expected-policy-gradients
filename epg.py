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
from agents.epg import EPG

from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', required=True, type=str,
                    choices=['cartpole','pendulum', 'cheetah'])
parser.add_argument('--quadrature', type=str, default='riemann',
                    choices=['riemann','trapz'])

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

def learn(env, config, quadrature, num_episodes = 5000, num_eval_final = 50, seed = 7):
    """
    Apply procedures of training for a DDPG.
    """

    env.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)

    config.batch_size = 100

    # initialize
    agent = EPG(env, config, quadrature=quadrature)
    agent.initialize()

    if not agent.discrete:
        print("Computing integral using '{}' method.".format(quadrature))

    # record one game at the beginning
    # if agent.config.record:
    #     agent.record()
    # model
        
    # Evaluate untrained policy
    evaluations = [agent.evaluate_policy()] 

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True 

    update_actor_freq = 100
    observation_list = []

    stats = {}
    stats["episode_rewards"] = []
    stats["evaluation_rewards"] = []
    while episode_num < num_episodes: #total_timesteps < config.max_timesteps:
        
        if done: 

            if total_timesteps != 0: 
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(total_timesteps, episode_num, episode_timesteps, episode_reward))
            
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

        observation_list.append(observation)
        # update policy
        #if total_timesteps%max(update_actor_freq, 10) == 0:
        if total_timesteps%update_actor_freq == 0:
            #update_actor_freq -= 1
            stats = agent.train_actor(observation_list)
            observation_list = []

        # Select action randomly or according to policy
        if total_timesteps < 0:#config.start_timesteps:
            action = env.action_space.sample()
        else:
            action, _ = agent.act(np.array(observation))

        if not agent.discrete:
            input_action = np.clip(action, agent.action_low, agent.action_high)
        else:
            input_action = action

        # Perform action
        new_observation, reward, done, _ = env.step(input_action) 
        done_bool = False if episode_timesteps + 1 == env._max_episode_steps else done
        episode_reward += reward
        action = np.array([action])
        agent.train_critic(observation, action, reward, new_observation, done_bool)

        agent.update_targets()

        observation = new_observation

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1


        
    # Final evaluation 
    evaluations.append(agent.evaluate_policy(eval_episodes=num_eval_final))


    # if agent.config.record:
    #     agent.record()



if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config(args.env_name)
    env = gym.make(config.env_name)
    # train model
    learn(env, config, args.quadrature)