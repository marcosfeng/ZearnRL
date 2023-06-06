//
// This Stan model defines a reinforcement learning model
// using Actor-Critic and applies it to the data.
//
data {
  int<lower=1> N;  // Number of subjects
  int<lower=1> T;  // Maximum number of rounds
  int<lower=1> S;  // Number of states
  int<lower=2> C;  // Number of choices
  array[N] int<lower=1, upper=T> Tsubj;  // Rounds total for each subject
  array[N,T,C] int<lower=0,upper=1> choice;  // Choice for each component (as binary values)
  array[N, T] real outcome;  // log badges
  array[N, T, S] real state;  // State Variables
  array[N,T] int<lower=0> week; // Number of the week
  int<lower=1> number_teachers; // Number of unique teachers
  array[N] int<lower=1, upper=number_teachers> group; // Teacher ID for each classroom

}
parameters {
  array[number_teachers, (C - 1)] real<lower=0, upper=1> cost;  // cost in badge-units for each component
  real<lower=0, upper=1> gamma[number_teachers];  // discount rate
  vector<lower=0, upper=1>[2] alpha[number_teachers];  // step-sizes
  real<lower=0> sensi[number_teachers];    // reward sensitivity
  real mu_cost;
  real<lower=0> sigma_cost;
  real mu_gamma;
  real<lower=0> sigma_gamma;
  vector[2] mu_alpha;
  vector<lower=0>[2] sigma_alpha;
  real mu_sensi;
  real<lower=0> sigma_sensi;
}

model {
  // hyperparameters
  mu_cost ~ normal(0.5, 1);
  sigma_cost ~ normal(0, 1);
  mu_gamma ~ beta(2, 2);
  sigma_gamma ~ normal(0, 1);
  mu_alpha ~ beta(2, 2);
  sigma_alpha ~ normal(0, 1);
  mu_sensi ~ normal(1, 1);
  sigma_sensi ~ normal(0, 1);

  // group-level parameters
  for (g in 1:G) {
    for (j in 1:(C - 1)) {
      cost[g, j] ~ normal(mu_cost, sigma_cost) T[0,1];
    }
    gamma[g] ~ normal(mu_gamma, sigma_gamma) T[0,1];
    alpha[g] ~ normal(mu_alpha, sigma_alpha) T[0,1];
    sensi[g] ~ normal(mu_sensi, sigma_sensi) T[0,];
  }

  // subject loop and trial loop
  for (i in 1:N) {
    // Save histories of w's and theta's
    matrix[S,C] w;      // State-Value weights
    matrix[S,C] theta;  // Policy weights
    real delta;
    array[C] real PE; // prediction error for each of the four components

    w = rep_matrix(0.0, S, C);
    theta = rep_matrix(0.0, S, C);

    for (t in 1:Tsubj[i]) {
      //Assign current state to a temporary variable
      row_vector[S] current_state = to_row_vector(state[i, t]);

      //Find choice probability
      for (j in 1:C) {
        choice[i, t, j] ~ bernoulli_logit( dot_product(theta[, j], current_state) );
      }

      if (t == Tsubj[i])
        continue; // Terminal State

      //Assign next state to a temporary variable
      row_vector[S] new_state = to_row_vector(state[i, t + 1]);

      // prediction error and update equations
      for (j in 1:(C - 1)) {
        if (choice[i, t, j] == 1) {
          PE[j] = gamma[group[i]]^(week[i, t + 1] - week[i, t])
                  * dot_product(w[, j], new_state)
                  - dot_product(w[, j], current_state);
          delta = sensi[group[i]] * (outcome[i, t + 1] - cost[group[i], j]) - PE[j];
          // Update w:
          w[, j] += alpha[group[i], 1] * delta * to_vector(current_state);
          // Update theta:
          theta[, j] += alpha[group[i], 2] * delta * to_vector(current_state);
        }
      }
      if (choice[i, t, C] == 1) {
        PE[C] = gamma[group[i]]^(week[i, t + 1] - week[i, t])
                * dot_product(w[, C], new_state)
                - dot_product(w[, C], current_state);
        delta = sensi[group[i]] * outcome[i, t + 1] - PE[C];
        // Update w:
        w[, C] += alpha[group[i], 1] * delta * to_vector(current_state);
        // Update theta:
        theta[, C] += alpha[group[i], 2] * delta * to_vector(current_state);
      }
    }
  }
}
