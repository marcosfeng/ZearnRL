# ZearnRL

ZearnRL is a project that uses Reinforcement Learning (RL) and Bayesian analysis to analyze and model Zearn data. The main file of this project is `zearn.qmd`, which contains the main code and documentation for the project.

## Main Files

1. **zearn.qmd**: This is the main file of the project. It contains the core code and logic for the reinforcement learning models. It uses the data from the 'Data' directory and the Python packages specified in the 'python-requirements.txt' file.

2. **Bayesian/code.R**: This file contains the code for performing Bayesian analysis. It uses the Stan files located in the 'Bayesian/Stan Files' directory and the 'cmdstanR' and 'brms' packages.

3. **Data**: This directory contains the data used in the project.

4. **python-requirements.txt**: This file lists all the Python packages used in the project.

## Stan Files

The Bayesian analysis uses several Stan files:

1. **Q-learning.stan**: This Stan model defines a Q-learning model and applies it to the data.

2. **Q-learning-states.stan**: This model is similar to the Q-learning model but includes states as a parameter.

3. **Q-learning-kernel.stan**: This model extends the Q-learning model by incorporating a kernel reward mechanism, which takes into account the similarity between state-action pairs.

4. **Q-kernel-hierarchical.stan**: This model extends the Q-learning-kernel model by adding a hierarchical structure, which allows for the modeling of multiple teachers with varying parameters.

5. **Actor-Critic.stan**: This model defines a reinforcement learning model using eligibility traces and applies it to the data.

## Data

The data used in this project is located in the `Data` directory. The directory contains `df_clean.csv`, but the raw data can be made available upon request.

## Python Packages

The Python packages required for this project are listed in the `python-requirements.txt` file. To install the required Python packages, run the following command:

```bash
pip install -r python-requirements.txt
```

## Usage

To run the code, first install the required packages. Then, run the `Bayesian/code.R` file for the Bayesian analysis and, finally, the `zearn.qmd` file for the analysis and main text.
