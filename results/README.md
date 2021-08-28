# Description of the results
This folder contains the results of different simulations in this work. The results are saved on ``pickle-file``. 

## Folder structure
Here, you find five different ``pickle-files``:
 - ``sim_results_fixed_target_15.pkl``: First results achieved in this work (outdated)
 - ``sim_results_fixed_target_15_Validation.pkl``: Results during testing phase, for the case ``offers_input.csv`` with Energy_target = 15 kWh for all episodes
 - ``sim_results_fixed_target_15_batch_improve.pkl``: Latest training results for the case ``offers_input.csv`` with Energy_target = 15 kWh for all episodes
 - ``training_results_30_partners.pkl``: Training results for the case ``offers_input_30_partners.csv``
 - ``training_results_30_partners_batch_improve.pkl``: Improved training results for the same case ``offers_input_30_partners.csv``, with the latest improvements in the training loop

## Results file description:
The results are organized in a python dictionary called ``data``:
 - ``data['agents']``: it has the number and label of the RL policies
 - ``data['simulation']``: it has the Energy_target, environment, number of episodes, number of steps per episode
 - ``data['outcome']``: it has the total reward, total regret, last step, Energy_target per episode
 - ``data['policy_sol']``: it has the actions, state, reward, regret Q-action function per number of steps
 - ``data['policy_dist']``: it has the final Q-action function per episode
