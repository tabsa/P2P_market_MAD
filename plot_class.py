## Class for the plots
# This files contains the plots (as class) used in this repository.
# They are used in the P2P_market_sim.file or in the Jupyter notebooks

#%% Import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

#%% Plot parameters
plt_colmap = plt.get_cmap("tab10", 15)
sns.set_style("whitegrid")

#%% Plot for the total reward over episodes
def plot_reward_per_episode(df_score, y_opt, plt_label, plt_marker, axis_label, plt_title):
    # Get the plot parameters
    no_RL_agents = len(df_score)
    no_episodes = df_score[0].shape[0]
    plt.figure(figsize=(10,7))
    x = np.arange(0, no_episodes)
    for i in range(no_RL_agents):
        # y-axis option
        if y_opt == 'total_reward':
            y = df_score[i]['total_rd']
        elif y_opt == 'gamma_rate':
            y = df_score[i]['total_rd'] / df_score[i]['final_step']  # gamma_per_epi (success rate)
        # plot option: matplotlib or plotly
        plt.plot(x,y, label=plt_label[i], marker=plt_marker[i], linestyle='--')
    # Legend and labels of the plot
    plt.legend(fontsize=16)
    plt.ylabel(axis_label[1], fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(axis_label[0], fontsize=16)
    plt.title(plt_title, fontsize=20)
    plt.show()

def plot_reward_distribution(df_score, plt_label, axis_label, plt_title):
    # Get the plot parameters
    no_RL_agents = len(df_score)
    plt_colmap = plt.get_cmap("tab10", no_RL_agents)
    plt.figure(figsize=(10,7))
    color = ['blue', 'green', 'orange']
    # For-loop for subplots
    for i in range(no_RL_agents):
        x = df_score[i]['final_state']
        y = df_score[i]['total_rd']
        #plt.subplot(no_RL_agents,1,i+1)
        plt.scatter(x, y, label=plt_label[i], c=color[i])
        # Legend and labels of the plot
        plt.legend(fontsize=16)
        plt.xlabel(axis_label[0], fontsize=16)
        plt.ylabel(axis_label[1], fontsize=16)
    plt.show()

def plot_action_choice(agent, axis_label, plt_title):
    plt.figure(figsize=(10,7))
    trials = np.arange(0, agent.env.no_trials)
    # Subplot 1
    plt.subplot(211)  # 2 rows and 1 column
    plt.scatter(trials, agent.action_n[0,:], cmap=plt_colmap, c=agent.action_n[0,:], marker='.', alpha=1)
    plt.title(plt_title[0], fontsize=16)
    plt.xlabel(axis_label[0], fontsize=16)
    plt.ylabel(axis_label[1], fontsize=16)
    plt.yticks(list(range(agent.env.no_offers)))
    plt.colorbar()
    # Subplot 2
    plt.subplot(212)
    plt.bar(trials, agent.action_n[1,:])
    #plt.bar(trials, self.state_n)
    plt.title(plt_title[1], fontsize=16)
    plt.xlabel(axis_label[0], fontsize=16)
    plt.ylabel(axis_label[2], fontsize=16)
    #plt.legend()
    plt.show()

def plot_regret_prob(regret_prob, epi_id, plt_label, axis_label, plt_title):
    # Plot regret over trial (opportunity cost of selecting a better action)
    agents = regret_prob.shape[0] # no RL agents
    plt.figure(figsize=(10, 7))
    # Subplot 1
    plt.subplot(211)  # 2 rows and 1 column
    for a in range(agents):
        plt.plot(np.cumsum(1 - regret_prob[a, epi_id, :]), label=plt_label[a])  # Plot per RL_agent
    plt.xlabel(axis_label[0], fontsize=16)
    plt.ylabel(axis_label[1], fontsize=16)
    plt.title(plt_title[0], fontsize=16)
    plt.legend()
    # Subplot 2
    plt.subplot(212)
    for a in range(agents):
        plt.plot(1 - regret_prob[a, epi_id, :], label=plt_label[a])
    plt.xlabel(axis_label[0], fontsize=16)
    plt.ylabel(axis_label[1], fontsize=16)
    plt.title(plt_title[1], fontsize=16)
    plt.legend()
    plt.show()