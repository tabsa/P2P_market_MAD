# Description of the input data
This Readme file explains the description of the csv-files in this ``input_data/`` folder.

## Folder structure
Here, you find two different csv-files:
 - File ``offers_input.csv`` with 15 offers
 - File ``offers_input_30_partners.csv`` with 30 offers

## CSV file description
Each row defines an available offer on the P2P market, which is characterized by 6 columns:
 - col ``id``: Integer number characterising the offer ID (starts in 0 because python indexes start at 0 as well)
 - col ``energy``: Float number with the energy quantity offered on the market
 - col ``price``: Float number with the offering price
 - col ``distance``: Euclidean distance, having the RL agent as the centric point (float number)
 - col ``co2``: CO2 emission (float number)
 - col ``sigma``: Probability of success for the Bernoulli distribution (float number between 0 and 1)
