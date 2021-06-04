## RL agent for the Energy P2P market as Multi-Armed Bandit (MAD) problem
# Application of MAD problem to the energy P2P market
# TBA

#%% Import packages
import numpy as np
import pandas as pd
import pickle as pkl
import os
# Import Class and functions
from MAD_env import trading_env, trading_agent
from plot_class import *

#%% Hyperparameters for the validation
## Path of the file with the training loop
wk_dir = os.getcwd() + '/results/' # Get from the 'results/' folder
train_file = 'sim_results_fixed_target_15.pkl' # Replace by other if you want
train_file = os.path.join(wk_dir, train_file)
out_filename = 'sim_results_fixed_target_15_Validation.pkl'
out_filename = os.path.join(wk_dir, out_filename)

#no_steps = policy.shape[1] # per episode
no_steps = 40 # It can be the same or not from the training loop...up to the user
no_episodes = 100 # Episodes for validation
training_epi = np.arange(90, 100) # episodes id to get the final optimal policy from the training loop

# Different types of energy_target scenarios:
target_bounds = np.array([5, 35]) # Random (and independent) between episodes
target_sample = np.random.uniform(low=target_bounds[0], high=target_bounds[1], size=no_episodes)
#target_bounds = np.arange(start=5, stop=51, step=5) # Progressive e_target = [5, 50]
#target_sample = np.repeat(target_bounds, 20)

## Output data
agent_list = [] # List with all RL agent
outcome_agent = [] # List of outcome DF per RL agent
# Name of the elements for the outcome_agent DF
df_col_name = ['mean_rd', 'final_step', 'energy_target', 'final_state',
               'avg_Q_val', 'std_Q_val', 'mean_regret', 'avg_reg_val', 'std_reg_val']
policy_agent = [] # List of policy solutions (array) per RL agent
policy_distribution = [] # List of estimator Q_val(arm_j) per RL agent
#######################################################################################################################

#%% Main Script
if __name__ == '__main__':
    ## Load the file with the training loop
    data = pkl.load(open(train_file, 'rb'))  # Read file
    env = data['simulation']['environment'] # Load the same env
    env.reset() # Important to Reset before using it again
    env.sample_seed = target_sample # Replace the energy target for the Validation phase

    ## Upload the parameters used on the training loop
    no_RL_agents = data['agents']['no'] # Same no and Policy per agent
    no_offers = env.no_offers
    policy_name = data['agents']['id'] # Upload the same Policies from the training loop
    optimal_policy = data['policy_dist'] # Optimal policy per agent, it is on a List
    policy_sol_epi = np.zeros((6, no_steps, no_episodes))  # Array to store policy solutions per episode
    policy_estimate_dist = np.zeros((no_episodes, no_offers, 3)) # 3 values stored per epi and offer, that is why we have a 3rd dimension
    for ag in range(no_RL_agents):
        agent = trading_agent(env, target_bounds, policy_name[ag], e_greedy=1) # Call the trading_agent class
        print(f'Run the agent {agent.policy_opt} with fixed Q-value(j):')  # Print which RL_agent by its policy_opt
        e = 0 # episode id
        while True:
            # Set the optimal policy
            agent.Arm_Q_j = optimal_policy[ag][training_epi, :, 0].mean(axis=0) # Q-value of each arm_j
            agent.N_arm = optimal_policy[ag][training_epi, :, 1].mean(axis=0) # No of times each arm_j was selected
            agent.thom_var = optimal_policy[ag][training_epi, :, 2].mean(axis=0) # Only for Thompson-Sampler (Variance of the Beta Dist)
            print(f'Episode {e} - Energy target {target_sample[e]}') # Episode print
            env.run(agent, e)  # Run environment, as inputs - RL_agent and epi_id
            # Store final results in np.arrays
            policy_sol_epi[:, :, e] = agent.policy_sol
            policy_estimate_dist[e, :, :] = agent.Q_val_final
            # Go to the next episode
            if e < no_episodes-1: # stopping condition
                e += 1
                # Reset of both agent and environment
                agent.reset()
            else: # Stop the loop
                break
        # end episodes for agent ag
        # Store the outcome parameters:
        outcome_agent.append(pd.DataFrame(agent.outcome, columns=df_col_name))
        policy_agent.append(policy_sol_epi)
        policy_distribution.append(policy_estimate_dist)
        # Reset the array for next agent in agent_list:
        policy_sol_epi = np.zeros((6, no_steps, no_episodes))
        policy_estimate_dist = np.zeros((no_episodes, no_offers, 3))
        print('\n')
    print(f'Validation phase is done')

    ## Save simulation results
    # Build a dictionary
    validation = {} # Results from the validation phase, including the Estimator computed from the Training loop
    # Validation info
    validation['episodes'] = training_epi
    validation['Arm_Q_j'] = [optimal_policy[i][training_epi, :, 0].mean(axis=0) for i in range(3)]
    validation['N_arm'] = [optimal_policy[i][training_epi, :, 1].mean(axis=0) for i in range(3)]
    validation['Thom_var'] = [optimal_policy[i][training_epi, :, 2].mean(axis=0) for i in range(3)]
    validation['episodes'] = no_episodes
    validation['steps'] = no_steps
    data['validation'] = validation # Assign the new 'validation' field to the data.dictionary
    # Update previous fields of the data dictionary
    data['simulation']['target'] = target_sample # Energy_target used for the validation
    data['outcome'] = outcome_agent
    data['policy_sol'] = policy_agent # Optimal policy per agent and episode
    data['policy_dist'] = policy_distribution  # Distribution of Q_val per arm_j (partner)
    file = open(out_filename, 'wb')
    pkl.dump(data, file)
    file.close()
