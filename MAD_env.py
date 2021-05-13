## Class to implement the P2P market as MAD problem
# Here we consider an episodic environment - as the carpole example
# Agent has fixed no.steps to achieve the desired energy target (per episode)
# The episode is done when the energy target is reached: final_state == energy_target
#   - we get a positive episode, meaning that total_reward is == 1 (the closest to One)
#   - Otherwise the episode fails when: no_steps == N && final_state < energy target
#
# Policy that maximizes the Action-value function (Q-fun) is by default reaching the highest total_reward
# Stopping condition (OR condition):
# 1. Agent reaches the energy target: final_state == energy_target
# 2. Episode reaches the last trial: no_steps == N
#
# As class of the code we have the following:
# class trading_env - Defines the environment of individual prosumer trading in a P2P market
# class trading_agent - Defines the RL_agent that interacts with the trading_env, responsible for using a specific policy

#%% Import packages
import numpy as np
import random as rnd
import pandas as pd
from scipy.stats import beta
from collections import deque # Double queue that works similar as list, and with faster time when using append

#%% Build the Environment
class trading_env:
    def __init__(self, no_offers, no_steps, offer_file, sample_type=None, sample_seed=None, no_preferences=2):  # Define parameters of the environment
        # time-space parameters
        self.no_steps = no_steps  # Time-step of the energy P2P market in the RL framework, duration of each episode [1,...,no_trials]
        self.end_trial = 0
        self.env_size = (no_offers, no_steps)
        # Offers parameters
        self.no_offers = no_offers
        self.offers_id = np.arange(no_offers)
        self.offer_file = offer_file
        self.sample_type = sample_type # Indicates if the sampling is as external input, or generated by the env
        self.sample_seed = sample_seed # Sample seed, here we just consider as trading_agent.energy_target
        self.offers_info = np.zeros((no_offers, 3))
        self.preference = np.zeros((no_offers, no_preferences))
        self.sigma_n = np.zeros((no_offers, no_steps))
        # Environment parameters
        self.env_simulation = 0  # Flag indicating the simulation is completed
        self.is_reset = False

    def reset(self): # Reset env so that we use for another episode
        self.end_trial = 0
        self.env_simulation = 0
        self.sigma_n = np.zeros((self.no_offers, self.no_steps))
        self.is_reset = True

    def run(self, trading_agent, epi): # Run the trading_env
        # Run the simulation, environment response, to the agent action per time_t [1,...,no_trials]
        if self.sample_type == 'External_sample':
            trading_agent.energy_target = self.sample_seed[epi] # seed of episode id (epi)
        else:
            trading_agent.energy_target = trading_agent.profile_sampling()  # Energy value as Target, where market.offers have reach from [1,...,no_trials]
        target_T = trading_agent.energy_target
        self.offers_sample(target_T)
        for n in range(self.no_steps):  # per step_n (id_n)
            trading_agent.id_n = n  # id_n equals to step_n of env.run. We synchronize internal id_n with environment loop.
            #### Agent choice in step n ####
            # Agent makes action in trial_n - action_n
            action_n = trading_agent.action()  # Calls action function of agent_class
            #### Environment update (state_n, reward_n) for step n ####
            # Binomial distribution sets signal {0,1}, where probability is defined by the expected revenue per var_n (each var_n has a exp_rev that represents the prob of success)
            trading_agent.reward_n[n] = np.random.binomial(1, p=self.offers_info[action_n,2])
            # Update state_n and action_n
            state_n_1 = trading_agent.state_n[n - 1] if n > 0 else 0 # state_n-1
            trading_agent.action_n[:, n], trading_agent.state_n[n] = self.update_state_reward(action_n, state_n_1, trading_agent.reward_n[n], target_T)
            trading_agent.total_reward += trading_agent.reward_n[n] # Total reward over n steps
            trading_agent.update_value_fun()  # Update of regret probability
            # Termination condition
            if target_T == trading_agent.state_n[n] or n == self.no_steps-1: # E_target is reached OR final step_n is reached
                self.env_simulation = 1
                self.end_step = n
                break
        # Collect the output of the agent
        trading_agent.collect_data()
        # Return of this function
        return self.env_simulation

    def update_state_reward(self, action_n, state_n_1, reward_n, target): # Update state_n (based on state_n-1 and action_n)
        # Update state_n and reward_n
        energy_n = self.offers_info[action_n, 0] if reward_n == 1 else 0 # E_n = E_j(a_n) if R_n == 1 Else E_n = 0 (Agent trades energy with offer j selected in action_n)
        energy_n = (target - state_n_1) if state_n_1 + energy_n > target else energy_n # In case we are in the final_step

        action_n = np.array([action_n, energy_n]) # Organize action_n as array
        state_n = state_n_1 + action_n[1]
        return action_n, state_n

    def offers_sample(self, target): # Function to sample offers [1,...,j] per trial_n
        # Energy offering sampling - Depends on the target (energy from trading_agent)
        ## Run the Sampling method - Input file, Monte Carlo, real-time, etc
        if self.sample_type == None or self.sample_type == 'External_sample':
            # Read the csv-file with all offer info
            offer_data = pd.read_csv(self.offer_file)  # Read the csv-file with all offer info
            self.offers_info = offer_data[['energy', 'price', 'sigma']].values
            self.preference = offer_data[['distance', 'co2']].values  # Preference (Distance and CO2)
        elif self.sample_type == 'Monte-Carlo': # Correct it - OLD VERSION
            # Monte-Carlo sampling - Assuming we change offer_energy per episode_id (time)
            offer_rnd_sample = np.random.sample(size=self.no_agents) # Random sample distribution per agent_j
            offer_rnd_sample = offer_rnd_sample/sum(offer_rnd_sample) # Get the ratio to distribute as Pro-rata
            self.energy = offer_rnd_sample*target
            # Price offering sampling - Assuming we have variance price per episode_id (time)
            for j in range(self.no_agents): # For-loop per agent_j
                j_mean = self.price_bounds[j, 0] # Mean price per agent j
                j_std = self.price_bounds[j, 1] # Std dev price per agent j
                self.price[j,:] = np.random.normal(loc=j_mean, scale=j_std, size=self.price.shape[1])

#%% Build the RL_agent and represented prosumer_i
class trading_agent: # Class of RL_agent to represent the prosumer i
    def __init__(self, env, e_target_bd, policy_opt, prosumer_type=None, no_sample=None, e_greedy=0.90):
        # Parameters - Environment info
        self.env = env  # Call the trading_env class we build above
        self.sim_shape = (env.no_offers, no_sample)  # Full shape of the simulation - action_size x no_episodes (no_samples)
        # Parameters - RL_Agent info
        self.is_reset = False
        self.is_exp_replay = False
        self.memory = deque(maxlen=1000) # Creates an empty queue with max lenght of 1000, it will work as memory of the NN
        self.prosumer_type = prosumer_type
        self.policy_opt = policy_opt # String with the policy strategy
        self.no_sample = no_sample # Number of samples for the simulation
        self.e_greedy = e_greedy # Probability for exploiting (epsilon_greedy)
        self.ep = np.random.uniform(0, 1, size=env.no_steps) # rand.prob for exploiting
        self.total_reward = 0
        self.action_choice = 0 # Action decision to be used every step_n
        self.multi_play_k = np.zeros(self.env.no_offers) # K times partner j (K_j) is selected in the same episode
        self.id_n = 0 # Index of step_n
        # Store info over step_n (action_t, state_t, reward_t)
        self.state_n = np.zeros(env.no_steps) # Settle energy for each step_n
        self.action_n = np.zeros((2, env.no_steps)) # Arrays 2 x no_steps
        self.reward_n = np.zeros(env.no_steps)
        self.Q_val_n = np.zeros(env.no_steps) # Q-action value per step_n
        self.Regret_n = np.zeros(env.no_steps) # Regret-action value per step_n (Or loss function)
        # Store info over action space [1,...,action_size], REMEMBER action_size is equal to the number of offers
        self.N_arm = np.zeros(env.no_offers) if self.policy_opt != 'Thompson_Sampler_policy' else np.ones(env.no_offers)
        self.Arm_Q_j = np.zeros(env.no_offers) if self.policy_opt != 'Thompson_Sampler_policy' else np.ones(env.no_offers)
        self.thom_var = np.ones(env.no_offers)
        # self.Arm_Q_j: Array with Q-action value for each offer_j (arm) updated every step_n. It estimates the Probability distribution associated with each offer_j
        # self.N_arm: Array with the No of times offer_j (arm) was chosen updated every step_n
        # self.thom_var: Array to estimate the variance of the Beta distribution for the Thompson-Sampler policy
        # Parameters - Prosumer info
        self.energy_target_bounds = e_target_bd # Max and Min energy per target_time. Energy_target <0 Consumer; >0 Producer
        # Output data
        self.outcome = []
        self.policy_sol = np.zeros((6, env.no_steps))
        self.data = None  # DataFrame storing the final info of each episode simulation

    def reset(self): # Reset operation after episode is done
        self.env.reset()
        self.action_choice = 0
        self.total_reward = 0
        self.energy_target = 0
        self.id_n = 0
        self.multi_play_k = np.zeros(self.env.no_offers)
        self.action_n = np.zeros((2, self.env.no_steps))
        self.reward_n = np.zeros(self.env.no_steps)
        self.state_n = np.zeros(self.env.no_steps)
        self.Q_val_n = np.zeros(self.env.no_steps)
        self.Regret_n = np.zeros(self.env.no_steps)
        self.ep = np.random.uniform(0, 1, size=self.env.no_steps)
        if not self.is_exp_replay:
            self.N_arm = np.zeros(self.env.no_offers) if self.policy_opt != 'Thompson_Sampler_policy' else np.ones(self.env.no_offers)
            self.thom_var = np.ones(self.env.no_offers)
            self.Arm_Q_j = np.zeros(self.env.no_offers) if self.policy_opt != 'Thompson_Sampler_policy' else np.ones(self.env.no_offers)
        self.is_reset = True

    def collect_data(self):  # Function to manipulate data into DataFrames
        # Get statistical results
        end_n = self.id_n # Final step_n
        mean_rd = self.Q_val_n[end_n] # mean reward of the episode, Remember Q_value computes the mean_rd every step_n, the final step_n gives us the estimated mean_rd of the episode
        avg_Q_n = self.Q_val_n[0:end_n].mean() # It can be confusing the name, but is the average Q_value over steps
        std_Q_n = self.Q_val_n[0:end_n].std() # Std dev of the Q_value
        mean_regret = self.Regret_n[end_n]
        avg_reg_n = self.Regret_n[0:end_n].mean()
        std_reg_n = self.Regret_n[0:end_n].std()
        # Organize output data
        self.outcome.append([mean_rd, end_n+1, self.energy_target, self.state_n[end_n],
                             avg_Q_n, std_Q_n, mean_regret, avg_reg_n, std_reg_n])
        self.policy_sol = np.vstack((self.action_n, self.state_n, self.reward_n, self.Q_val_n, self.Regret_n)) # Policy per step_n
        self.Q_val_final = np.dstack((self.Arm_Q_j, self.N_arm, self.thom_var)) # Q-action value per offer_j

    def profile_sampling(self): # Function to generate/collect the energy profile of prosumer_i
        ## Run a Uniform sampling - Single value but it can be a time-series
        return np.random.uniform(low=self.energy_target_bounds[0], high=self.energy_target_bounds[1])

    def update_value_fun(self): # Function to update the Q_fun and regret arrays over step_n and partner j
        n = self.id_n # step n
        j_arm = self.action_choice # action n
        # Update the Value function for each partner j. It is updated everytime partner j is selected by action_n
        # self.N_arm counts the no of times partner j was selected
        self.N_arm[j_arm] += 1

        # When self.policy_opt --> 'Thompson-Sampler', we have to calculate the Beta variance as well:
        # self.thom_var: increments 1 when partner_j has reward = 0 (we miss). This way the Beta distribution is spreaded so more 'stochastic'
        # It is like the ratio of rd_j / N_j drops everytime we miss revenue in j. Increase the change of another partner being selected later on
        self.thom_var[j_arm] += 1 - self.reward_n[n] if self.policy_opt == 'Thompson_Sampler_policy' else 0

        ##############################################################################################################
        # Update the Q-Value for the selected partner j (Estimated mean prob mu_j^n)
        self.Arm_Q_j[j_arm] += (1 / self.N_arm[j_arm]) * (self.reward_n[n] - self.Arm_Q_j[j_arm])
        # Count the successful times partner j was selected (only valid for the same episode)
        self.multi_play_k[j_arm] += 1 if self.reward_n[n] == 1 else 0 # Increments 1 everytime Rd_n(j) is 1 (success offer j)

        # Calculate for step_n the Q-value function (that quantifies the expected return):
        Qval_n_1 = self.Q_val_n[n-1] if n>0 else 0 # n==0 we have 0
        self.Q_val_n[n] = Qval_n_1 + 1/(n+1) * (self.reward_n[n] - Qval_n_1)
        # The opportunity cost (regret) depends on NOT exploiting others 'non-seen' (less prob) arm_j
        Reg_n_1 = self.Regret_n[n-1] if n>0 else 0
        self.Regret_n[n] = Reg_n_1 + 1/(n+1) * (np.max(self.Arm_Q_j) - self.Arm_Q_j[self.action_choice] - Reg_n_1)

    def action(self):  # Function to make the action of the agent over step_n
        multi_play = self.multi_play_k >= 3
        if self.policy_opt == 'Random_policy':
            # Algorithm with random choice
            self.action_choice = self.rand_policy(multi_play)
        elif self.policy_opt == 'e-greedy_policy':
            # Algorithm with e-greedy approach
            self.action_choice = self.eGreedy_policy(multi_play)
        elif self.policy_opt == 'Thompson_Sampler_policy':
            # Algorithm with Thompson Sampler approach
            self.action_choice = self.tpSampler_policy(multi_play)
        # Return the result
        return self.action_choice

    def exp_replay(self, batch_size, greedy=True): # Training the theta per action over episodes e
        ## Per episode c, we update the var_theta so that we can have a better guess for action_n
        # mini_batch has all episodes i in the memory - Selected randomly
        mini_batch = rnd.sample(self.memory, batch_size)
        # Arm_Q_j for episode c - Prob as weighted average per final_state and gamma_bth
        theta_rd_c = np.zeros(self.env.no_offers) # self.a - sum of rewards per action_n
        theta_at_c = np.zeros(self.env.no_offers) # self.b - no. times action_n was selected
        thom_var_mu = np.zeros(self.env.no_offers)  # self.thom_var - Variance of the Beta distribution (Thompson-Sampler policy)
        total_state = 0
        # For-loop to calculate var_theta per episode i in mini_batch
        for theta_rd_i, theta_at_i, thom_var_i, total_rd_i, final_step_i, final_state_i in mini_batch: # Get elements per deque (episode i)
            gamma_i = total_rd_i / final_step_i # gamma per episode i
            # Weighted sum
            theta_rd_c += theta_rd_i if greedy else final_state_i * gamma_i * theta_rd_i
            theta_at_c += theta_at_i if greedy else final_state_i * gamma_i * theta_at_i
            thom_var_mu += thom_var_i if greedy else final_state_i * gamma_i * thom_var_i
            total_state += 1 if greedy else final_state_i
        # Get the average result by dividing for total_state
        #theta_rd_c = theta_rd_c / total_state  # avg_sum_reward per action_n - avg_a
        theta_c = theta_rd_c / total_state  # avg_sum_reward per partner j - avg_var_theta
        theta_at_c = theta_at_c / total_state  # avg_no.times per partner j - avg_b
        thom_var_mu = thom_var_mu / total_state # avg variance per partner j - avg_thom_var
        theta_c = np.nan_to_num(theta_c, nan=0)

        # Return the final var_theta_avg
        self.N_arm = theta_at_c #.round(1)
        self.Arm_Q_j = theta_c #.round(1)
        self.thom_var = thom_var_mu.round(0)
        if self.policy_opt == 'e-greedy_policy':
            self.e_greedy = 1 # Pure greedy but other we could have a decay based on the episode id - IMPROVE

        self.is_exp_replay = True

    def rand_policy(self, mask): # Implements the random choice algorithm
        mask_offers = self.env.offers_id[np.invert(mask)]
        return int(np.random.choice(mask_offers))

    def eGreedy_policy(self, mask): # Implements the e-greedy algorithm to explore_exploit
        mask_offers = self.env.offers_id[np.invert(mask)]
        mask_theta = np.ma.array(self.Arm_Q_j, mask=mask)

        # Step that controls IF we choice Exploration or Exploitation
        # If ep_prob[id_n] < e_greedy: best action, Else: Random_choice
        aux = np.argmax(mask_theta) if self.ep[self.id_n] <= self.e_greedy else np.random.choice(mask_offers)
        # Return the result of the function
        return int(aux)

    def tpSampler_policy(self, mask):
        # Thompson Sampler policy
        mu = self.N_arm * self.Arm_Q_j # mu = \sum(Reward[n] for all j)
        var = self.thom_var # var = \sum(1 - Reward[n] for all j)
        # Sampling a probability based on distribution (We assumed Beta Bernoulli) based on cumulative reward (self.a) and no. selected-times (self.b) per var_n
        # The Beta-dist captures the uncertainty per var_n that is changing over time_t...
        # Per time_t indicates the probablity of getting reward per var_n, it is dynamic over trials until the Beta-dist goes towards the var_n with highest reward
        # Rule of thumb - largest mean and smallest std.dev results in greater prob of being selected...
        # ...var_n with low revenue is expected a wider distribution with small mean and large std.dev resulting in high uncertainty for future action choice
        sample_theta = np.random.beta(mu, var)
        mask_theta = np.ma.array(sample_theta, mask= mask)
        # Select the var_n (machine) with highest expected revenue (based on cumulative probability over time_t)
        return int(np.argmax(mask_theta))
