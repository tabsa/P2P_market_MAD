# Peer-to-Peer Energy Market solved by Reinforcement Learning 
This repository contains the code and presentation to reproduce all the figures and experiments presented in the 2020 INFORMS Annual meeting:

T. Sousa and P. Pinson, Fairness-Driven Agent for P2P Electricity Markets: Reinforcement Learning Application. https://www.youtube.com/watch?v=pIWMP_kvS48

## Dependencies
The code is written in Python 3.8 and uses the libraries in the [requirements file](https://github.com/tabsa/P2P_market_MAD/blob/main/requirements.txt).
Command for creating a virtual environment:
 - ```conda create --name <env_name> --file requirements.txt```

## Repository structure
This repository has the following directory structure
- Main folder: scripts, notebooks and requirements.txt
- input_data folder: CSV-files with the input data used in this work, with a README file
- results folder: Pickle-files storing the results of different simulations, with a README file
- Presentations folder: PDF-files with presentations of this work

## Reproduce experiments with the scripts
Execute the following python scripts, which are grouped by task.
 - Training the RL agent for a particular case 
    - Open file [training_loop.py](https://github.com/tabsa/P2P_market_MAD/blob/main/training_loop.py), and change the case csv.file and Hyperparameters
    ```
    #%% Hyperparameters for the training
    # Input parameters
    input_dir = os.getcwd() + '/input_data/' # Define other if you want
    in_filename = 'offers_input.csv'
    in_filename = os.path.join(input_dir, in_filename) # Path for the Input csv.file
    no_steps = 40 # per episode
    no_episodes = 100
    no_RL_agents = 3 # each agent has a different policy
    batch_size = 20 # exp replay buffer, also dictates the episodes for training and testing
    # Basically, batch_size dictates the first 20 episodes are for pure exploration, while the exploitation starts on the remaining ones

    ## RL_agent policies (to be simulated)
    agent_policy = ['Random_policy', 'e-greedy_policy', 'Thompson_Sampler_policy']
    
    ## Saving file
    wk_dir = os.getcwd() + '/results/' # Define other if you want
    out_filename = 'sim_results_fixed_target_15_batch_improve.pkl'
    out_filename = os.path.join(wk_dir, out_filename)
    ```
   - Run the file
   - Analyse the results in the Notebook [Data_analysis_and_Policy](https://github.com/tabsa/P2P_market_MAD/blob/main/Data_analysis_and_Policy.ipynb)
 - Testing the RL agent after training
    - Open file [P2P_simulation] (https://github.com/tabsa/P2P_market_MAD/blob/main/P2P_simulation.py)
    - Define the training pickle.file and Hyperparameters
  ```
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
  ```
   - Use the same Notebook for analysing the results


## Notebooks

## References
