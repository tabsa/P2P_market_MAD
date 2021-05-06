## RL agent for the Energy P2P market as Multi-Armed Bandit (MAD) problem
# Application of MAD problem to the energy P2P market
#

#%% Import packages
import numpy as np
import pandas as pd
import pickle as pkl
import os
# Import Class and functions
from MAD_env import trading_env, trading_agent
from plot_class import *

#%% Hyperparameters for the simulation
## Simulation
no_steps = 40 # per episode
no_episodes = 100
no_RL_agents = 3 # each agent has a different policy
batch_size = 20 # exp replay buffer

## RL_agent policies (to be simulated)
agent_policy = ['Random_policy', 'e-greedy_policy', 'Thompson_Sampler_policy']

## P2P market
no_offers = 15 # no of offers the RL_agent/prosumer can trade with
# Different types of energy_target scenarios:
#target_bounds = np.array([3, 25]) # Random (and independent) between episodes
#target_sample = np.random.uniform(low=target_bounds[0], high=target_bounds[1], size=no_episodes)
target_bounds = 15 # Fixed e_target
target_sample = target_bounds * np.ones(no_episodes)
#target_bounds = np.arange(start=5, stop=51, step=5) # Progressive e_target = [5, 50]
#target_sample = np.repeat(target_bounds, 20)

## Output data
agent_list = [] # List with all RL agent
outcome_agent = [] # List of outcome DF per RL agent
# Name of the elements for the outcome_agent DF
df_col_name = ['mean_rd', 'final_step', 'energy_target', 'final_state', 'final_Q_val',
                'mean_Q_val', 'std_Q_val', 'final_G_regret', 'mean_G_regret', 'std_G_regret']
policy_agent = [] # List of policy solutions (array) per RL agent

## Saving file
wk_dir = os.getcwd() # Define other if you want
out_filename = 'sim_results_fixed_target_15_new_version.pkl'
out_filename = os.path.join(wk_dir, out_filename)
#######################################################################################################################

#%% Main Script
if __name__ == '__main__':
    ## Initialize arrays
    policy_sol_epi = np.zeros((6, no_steps, no_episodes))  # Array to store policy solutions per episode
    policy_distribution = []
    policy_estimate_dist = np.zeros((no_episodes, no_offers, 3)) # 3 values stored per epi and offer, that is why we have a 3rd dimension

    ## Create MAD environment and RL_agents
    env = trading_env(no_offers, no_steps, 'input_data/offers_input.csv', 'External_sample', target_sample)
    agent_list.append(trading_agent(env, target_bounds, agent_policy[0]))  # Random policy
    agent_list.append(trading_agent(env, target_bounds, agent_policy[1], time_learning=10, e_greedy=0.25))  # e-Greedy policy
    agent_list.append(trading_agent(env, target_bounds, agent_policy[2]))  # Thompson-Sampler policy

    ## Simulation phase
    ag = 0  # id of agent
    for agent in agent_list:  # For-loop per RL agent
        # Simulate the agent interaction
        print(f'Run the agent with the {agent.policy_opt}:')
        for e in range(no_episodes):  # For-loop per episode e
            # Episode print
            print(f'Episode {e} - Energy target {target_sample[e]}')
            if agent.is_reset or e == 0:
                env.run(agent, e)  # Run environment, inputs we have RL_agent and episode id
                # Store info in the memory
                agent.memory.append((agent.Arm_Q_j, agent.N_arm, agent.thom_var, agent.total_reward, agent.id_n, agent.state_n[agent.id_n]))
                if len(agent.memory) >= batch_size:  # and len(agent.memory) <= 50:
                    agent.exp_replay(batch_size, greedy=True)
                    batch_size += 1  # Increase the batch size of previous episodes, to propagate the long-term memory
                # Store final results in np.arrays
                policy_sol_epi[:, :, e] = agent.policy_sol
                policy_estimate_dist[e, :, :] = agent.Q_val_final
            # Reset of both agent and environment
            agent.reset()

        # Store the outcome parameters:
        outcome_agent.append(pd.DataFrame(agent.outcome, columns=df_col_name))
        policy_agent.append(policy_sol_epi)
        policy_distribution.append(policy_estimate_dist)
        # Reset the array for next agent in agent_list:
        policy_sol_epi = np.zeros((6, no_steps, no_episodes))
        policy_estimate_dist = np.zeros((no_episodes, no_offers, 3))
        # Next agent (1st For-loop)
        ag += 1
        batch_size = 20  # Reset batch_size
        print('\n')
    print(f'All {no_episodes} Episodes are done')

    #%% Save simulation results
    # Build a dictionary
    data = {}
    agents = {}
    simulation = {}
    # agents info
    agents['no'] = no_RL_agents
    agents['id'] = agent_policy
    # simulation info
    simulation['target'] = target_sample
    simulation['environment'] = env
    simulation['episodes'] = no_episodes
    simulation['trials'] = no_steps
    data['agents'] = agents
    data['simulation'] = simulation
    data['outcome'] = outcome_agent
    data['policy_sol'] = policy_agent
    data['policy_dist'] = policy_distribution
    # file = open(out_filename, 'wb')
    # pkl.dump(data, file)
    # file.close()
