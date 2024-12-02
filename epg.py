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

import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', required=True, type=str,
                    choices=['cartpole','pendulum', 'cheetah'])
parser.add_argument('--quadrature', type=str, default='riemann',
                    choices=['riemann','trapz'])
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--save_as', type=str, default='results')
parser.add_argument('--num_episodes', type=int, default=700)
parser.add_argument('--num_eval_final', type=int, default=50)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--record', action='store_true')
parser.add_argument('--eval_from_checkpoint', action="store_true")
parser.add_argument('--learn_std', action="store_true")

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

def learn(env, config, quadrature, num_episodes = 5000, num_eval_final = 50, seed = 7, run=0, record=False, learn_std=False):
    """
    Apply procedures of training for a DDPG.
    """

    config.batch_size = 100

    # initialize
    agent = EPG(env, config, quadrature=quadrature, run=run, learn_std=learn_std)
    agent.initialize()

    if not agent.discrete:
        print("Computing integral using '{}' method.".format(quadrature))

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

    update_actor_freq = 100
    observation_list = []

    stats = {}
    stats["episode_rewards"] = []
    stats["evaluation_rewards"] = []
    stats["grad_norms"] = [] # TODO
    stats["first_integrand"] = None
    stats["last_integrand"] = None
    stats["cummulative_timesteps"] = []
    stats["episode_timesteps"] = []
    stats["eval_episode_timesteps"] = []
    stats["eval_cummulative_timesteps"] = []
    stats["integral_action_values"] = None # TODO
    stats["loss_integral"] = []
    stats["update_actor_freq"] = update_actor_freq
    stats["num_episodes"] = num_episodes
    stats["num_eval_final"] = num_eval_final
    stats["seed"] = seed
    stats["agent"] = agent.agent_name
    stats["quadrature"] = quadrature
    stats["env"] = config.env_name
    stats["num_actor_updates"] = 0

    while episode_num < num_episodes: #total_timesteps < config.max_timesteps:
        
        if done: 

            if total_timesteps != 0: 
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(total_timesteps, episode_num, episode_timesteps, episode_reward))
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

        observation_list.append(observation)
        
        # update policy
        if total_timesteps%update_actor_freq == 0:
            stats = agent.train_actor(observation_list, stats=stats)
            stats["num_actor_updates"] += 1
            observation_list = []

        # Select action according to our policy
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
    rewards, eval_cummulative_timesteps, eval_episode_timesteps = agent.evaluate_policy(eval_episodes=num_eval_final)
    stats["evaluation_rewards"] = rewards
    stats["eval_cummulative_timesteps"] = eval_cummulative_timesteps
    stats["eval_episode_timesteps"] = eval_episode_timesteps

    agent.save_model()
    if record:
        agent.record()
    agent.close()

    return stats

    # if agent.config.record:
    #     agent.record()

def eval_from_checkpoint(env, config, quadrature, run=0):
    """
    Apply procedures of training for a DDPG.
    """

    agent = EPG(env, config, quadrature=quadrature, run=run)
    agent.initialize()

    agent.restore()
    agent.record()
    agent.close()



if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config(args.env_name)
    env = gym.make(config.env_name)
    
    seed = args.seed
    env.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

    if args.eval_from_checkpoint:
        for i in range(args.runs):
            # train model
            eval_from_checkpoint(env, config, args.quadrature, run=i)
    else:
        outfile = "{}.pickle".format(args.save_as)
        outpath = os.path.join(config.output_path, "epg-{}".format(args.quadrature), outfile)
        runs = {}
        for i in range(args.runs):
            # train model
            stats_dict = learn(env, config, args.quadrature, 
                                num_episodes=args.num_episodes, 
                                num_eval_final=args.num_eval_final, 
                                seed=seed, 
                                run=i,
                                record = args.record,
                                learn_std = args.learn_std)
            runs[i] = stats_dict
        # save dictionary 
        pickle_out = open(outpath,"wb")
        pickle.dump(runs, pickle_out)
        pickle_out.close()
        pickle_in = open(outpath,"rb")
        example_dict = pickle.load(pickle_in)
        print("Stored the following runs:")
        for key, val in example_dict.items():
            print("run {} | number of keys in stats dict: {}".format(key, len(val.keys())))
