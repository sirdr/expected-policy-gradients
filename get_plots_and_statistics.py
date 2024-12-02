
import os
import argparse
import numpy as np

import matplotlib.pyplot as plt


import gym
import os

import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', required=True, help="Base directory for storing run outputs")

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
def lineplotCI(ax, x_data, y_data, low_CI, upper_CI, label):
    # Plot the data, set the linewidth, color and transparency of the
    # line, provide a label for the legend
    ax.plot(x_data, y_data, lw = 1, color = '#539caf', alpha = 1, label = label)
    # Shade the confidence interval
    ax.fill_between(x_data, low_CI, upper_CI, color = '#539caf', alpha = 0.4)
    # Label the axes and provide a title


def get_results(results, save_dir=None):

    reward_trajectories = get_trajectories(results)
    reward_trajectories_stats = get_trajectory_statistics(reward_trajectories)

    eval_reward_trajectories = get_eval_trajectories(results)
    eval_statistics = get_eval_statistics(eval_reward_trajectories)

    for agent, stats in eval_statistics.items():

        print("Evaluation Statistics for {} over 50 episodes after training:".format(agent))

        print("Mean Evaluation Reward : {} +/- {} |  [{} , {}] 90% CI".format(stats["mean"], stats["std_mean"], stats["ci_lower_mean"], stats["ci_upper_mean"]))
        print("Std Evaluation Reward : {} +/- {}  |  [{} , {}] 90% CI".format(stats["mean_std"], stats["std_mean_std"], stats["ci_lower_std"], stats["ci_upper_std"]))
    
        # Save evaluation statistics to a text file
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stats_file = os.path.join(save_dir, "evaluation_statistics.txt")
        with open(stats_file, "w") as f:
            for agent, stats in eval_statistics.items():
                f.write(f"Evaluation Statistics for {agent} over 50 episodes after training:\n")
                f.write(f"Mean Evaluation Reward : {stats['mean']} +/- {stats['std_mean']} |  [{stats['ci_lower_mean']}, {stats['ci_upper_mean']}] 90% CI\n")
                f.write(f"Std Evaluation Reward : {stats['mean_std']} +/- {stats['std_mean_std']} |  [{stats['ci_lower_std']}, {stats['ci_upper_std']}] 90% CI\n\n")
        print(f"Saved evaluation statistics to {stats_file}")

    _, ax = plt.subplots()

    for agent, stats in reward_trajectories_stats.items():
        ci_lower, ci_upper = stats["ci_lower"], stats["ci_upper"]
        avg_rewards = stats["mean"]
        std_rewards = stats["std"]

        episodes = np.arange(len(avg_rewards))

        lineplotCI(ax, episodes, avg_rewards, ci_lower, ci_upper, agent)

    ax.set_title("")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Average Reward")

    # Display legend
    ax.legend(loc = 'best')

    # Save the plot
    if save_dir:
        plot_file = os.path.join(save_dir, "reward_trajectories.png")
        plt.savefig(plot_file)
        print(f"Saved plot to {plot_file}")

    # Show the plot (optional, can be removed if not needed)
    plt.show()






    

if __name__ == '__main__':

    args = parser.parse_args()
    base_dir = args.base_dir

    agent_results = {}

    for j, agent in enumerate(os.listdir(base_dir)):
        agent_path = os.path.join(base_dir, agent)
        # Skip non-directory entries
        if not os.path.isdir(agent_path):
            continue
        results = {}
        for i, run_dir in enumerate(os.listdir(agent_path)):
            run_path = os.path.join(agent_path, run_dir)
            # Check for results.pickle in the run's results folder
            results_path = os.path.join(run_path, "results", "results.pickle")
            if os.path.exists(results_path):
                try:
                    with open(results_path, "rb") as f:
                        stats = pickle.load(f)
                        results[i] = stats
                except Exception as e:
                    print(f"Error loading pickle file from {results_path}: {e}")
            else:
                print(f"No results.pickle file found in {run_path}/results/")
        agent_results[agent] = results

    print("Found the following result dictionaries:")
    for key, val in agent_results.items():
        print("agent: {} | dict: {}".format(key, [k for k in val.keys()]))

    get_results(agent_results, save_dir=base_dir)