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
    ```
   - Run the file
    - 


## Notebooks

## References
