
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

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', required=True, type=str,
                    choices=['cartpole','pendulum', 'cheetah'])

agents = ["epg-riemann", "epg-trapz", "td3ddpg"]

t = 2.132 # from t-distribution for confidence interval of 90% and dof of 4 (i.e. num_runs - 1)

def get_trajectory_statistics(trajectories):

    traj_stats = {}

    for agent, runs in trajectories.items():

        traj_array = []
        num_traj = 0
        for run, traj  in runs.items():
            traj_array.append(traj)
            num_traj += 1

        traj_array = np.array(traj_array)

        assert traj_array.shape[0] == num_traj

        mean_traj = np.mean(traj_array, axis = 0)
        std_traj = np.std(traj_array, axis  = 0)
        ci_upper = mean_traj + (2.132*std_traj/(np.sqrt(num_traj)))
        ci_lower = mean_traj - (2.132*std_traj/(np.sqrt(num_traj)))

        traj_stats[agent] = {"mean": mean_traj , "std": std_traj, "ci_upper": ci_upper, "ci_lower": ci_lower}
    return traj_stats

def get_eval_statistics(trajectories):

    traj_stats = {}

    for agent, runs in trajectories.items():

        traj_array = []
        num_traj = 0
        for run, traj  in runs.items():
            traj_array.append(traj)
            num_traj += 1

        traj_array = np.array(traj_array)

        assert traj_array.shape[0] == num_traj

        num_evals = traj_array.shape[1]

        mean_traj = np.mean(traj_array, axis = 1) # average over time (instead of runs)
        std_traj = np.std(traj_array, axis  = 1) # average over time (insted of runs)

        average_mean_traj = np.mean(mean_traj) # average over average of runs
        average_std_traj = np.mean(std_traj) # average over std of runs
        std_mean_traj = np.std(mean_traj)
        std_std_traj = np.std(std_traj)

        ci_upper_mean = average_mean_traj + (2.132*std_mean_traj/(np.sqrt(num_traj)))
        ci_lower_mean = average_mean_traj - (2.132*std_mean_traj/(np.sqrt(num_traj)))
        ci_upper_std = average_std_traj + (2.132*std_std_traj/(np.sqrt(num_traj)))
        ci_lower_std = average_std_traj - (2.132*std_std_traj/(np.sqrt(num_traj)))

        traj_stats[agent] = {"mean_array": mean_traj , 
                            "mean": average_mean_traj, 
                            "std_array":std_traj, 
                            "mean_std":average_std_traj,
                            "std_mean": std_mean_traj,
                            "std_mean_std":std_std_traj,
                            "ci_upper_mean": ci_upper_mean, 
                            "ci_lower_mean": ci_lower_mean,
                            "ci_upper_std": ci_upper_std, 
                            "ci_lower_std": ci_lower_std}
    return traj_stats

def get_eval_trajectories(results):
    trajectories = {}
    for agent, runs_dict in results.items():
        rewards = {}
        for run, stats in runs_dict.items():
            rewards[run] = stats["evaluation_rewards"]
        trajectories[agent] = rewards
    return trajectories

def get_trajectories(results):
    trajectories = {}
    for agent, runs_dict in results.items():
        rewards = {}
        for run, stats in runs_dict.items():
            rewards[run] = stats["episode_rewards"]
        trajectories[agent] = rewards
    return trajectories

# Function based on code from https://www.datascience.com/blog/learn-data-science-intro-to-data-visualization-in-matplotlib  
def lineplotCI(ax, x_data, y_data, low_CI, upper_CI):
    # Plot the data, set the linewidth, color and transparency of the
    # line, provide a label for the legend
    ax.plot(x_data, y_data, lw = 1, color = '#539caf', alpha = 1, label = 'Fit')
    # Shade the confidence interval
    ax.fill_between(sorted_x, low_CI, upper_CI, color = '#539caf', alpha = 0.4, label = '95% CI')
    # Label the axes and provide a title


def get_results(results, env, config):

    reward_trajectories = get_trajectories(results)
    reward_trajectories_stats = get_trajectory_statistics(reward_trajectories)

    eval_reward_trajectories = get_eval_trajectories(results)
    eval_statistics = get_eval_statistics(eval_reward_trajectories)

    for agent, stats in eval_statistics.items():

        print("Evaluation Statistics for {} over 50 episodes after training:".format(agent))

        print("Mean Evaluation Reward : {} +/- {} |  [{} , {}] 90% CI".format(stats["mean"], stats["std_mean"], stats["ci_lower_mean"], stats["ci_upper_mean"]))
        print("Std Evaluation Reward : {} +/- {}  |  [{} , {}] 90% CI".format(stats["mean_std"], stats["std_mean_std"], stats["ci_lower_std"], stats["ci_upper_std"]))

    _, ax = plt.subplots()

    for agent, stats in reward_trajectories_stats.items():
        ci_lower, ci_upper = stats["ci_lower"], stats["ci_upper"]
        avg_rewards = stats["mean"]
        std_rewards = stats["std"]

        episodes = np.arange(len(avg_rewards))

        lineplotCI(ax, episodes, avg_rewards)


    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Display legend
    ax.legend(loc = 'best')


# Call the function to create plot
lineplotCI(x_data = daily_data['temp']
           , y_data = fitted_values
           , sorted_x = CI_df['x_data']
           , low_CI = CI_df['low_CI']
           , upper_CI = CI_df['upper_CI']
           , x_label = 'Normalized temperature (C)'
           , y_label = 'Check outs'
           , title = 'Line of Best Fit for Number of Check Outs vs Temperature')






    

if __name__ == '__main__':

    args = parser.parse_args()
    config = get_config(args.env_name)
    env = gym.make(config.env_name)

    path = os.path.join("./", config.output_path)

    dirs = os.listdir(path)

    results = {}

    for d in dirs:
        if d in agents:
            stats = None
            temp_path = os.path.join(path, d, "results.pickle")
            try:
                pickle_in = open(temp_path,"rb")
                stats = pickle.load(pickle_in)
            except:
                print("Did not find pickle file '{}' in directory '{}'".format("results.pickle",d))
            results[d] = stats
    print("Found the following result dictionaries:")
    for key, val in results.items():
        print("agent: {} | dict: {}".format(key, [k for k in val.keys()]))

    get_results(results, env, config)