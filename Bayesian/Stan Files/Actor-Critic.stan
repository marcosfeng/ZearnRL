//
// This Stan model defines an Actor-Critic RL model
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
}
parameters {
  array[C] real<lower=0> cost;  // cost in badge-units for each component
  real<lower=0.6, upper=1> gamma;  // discount rate
  real<lower=0> tau;  // temperature
  vector<lower=0, upper=1>[2] alpha;  // step-sizes
  array[C] vector[S] w_0;  // initial Ws
  array[C] vector[S] theta_0;  // initial Thetas
}
model {
  // Flat-ish priors
  cost     ~ normal(0, 2);
  gamma    ~ normal(0.8, 0.1);
  tau      ~ normal(0, 5);
  alpha    ~ uniform(0, 1);
  for (c in 1:C) {
    w_0[c]      ~ normal(0, 2);
    theta_0[c]  ~ normal(0, 2);
  }

  // subject loop and trial loop
  for (i in 1:N) {
    // Save histories of w's and theta's
    array[C] vector[S] w = w_0;      // State-Value weights
    array[C] vector[S] theta = theta_0;  // Policy weights
    real delta;
    real PE; // prediction error for each of the four components

    for (t in 1:Tsubj[i]) {
      //Assign current state to a temporary variable
      row_vector[S] current_state = to_row_vector(state[i, t]);
      // if  (t == 10) {
      //   print("theta[j]: ", theta);
      //   print("current_state: ", current_state);
      //   print("tau: ", tau);
      // }
      //Find choice probability
      for (j in 1:C) {
        choice[i, t, j] ~ bernoulli_logit(tau * dot_product(theta[j], current_state));
      }
      if (t == Tsubj[i])
        continue; // Terminal State

      //Assign next state to a temporary variable
      row_vector[S] new_state = to_row_vector(state[i, t + 1]);
      // prediction error and update equations
      for (j in 1:C) {
        if (choice[i, t, j] == 1) {
          PE = gamma^(week[i, t + 1] - week[i, t])
               * dot_product(w[j], new_state)
               - dot_product(w[j], current_state);
          delta = (outcome[i, t] - cost[j]) + PE;
          // Update w:
          w[j] += alpha[1] * delta * to_vector(current_state);
          // Update theta:
          theta[j] += alpha[2] * delta * to_vector(current_state) * tau
                        ./ (1 + exp(dot_product(theta[j], current_state) * tau));
        }
      }
    }
  }
}
generated quantities {
  array[N, T, C] real choice_prob;  // Probability of choosing each component
  // For log likelihood calculation
  vector[N] log_lik;

  for (i in 1:N) {
    array[C] vector[S] w = w_0;  // Initialize at w_0
    array[C] vector[S] theta = theta_0;  // Initialize at theta_0
    real delta;
    real PE;  // Prediction error
    log_lik[i] = 0; // initialize log likelihood for each subject

    for (t in 1:Tsubj[i]) {
      row_vector[S] current_state = to_row_vector(state[i, t]);

      for (j in 1:C) {
        // Compute the choice probability
        choice_prob[i, t, j] = inv_logit( tau * dot_product(theta[j], current_state) );
        log_lik[i] += bernoulli_lpmf(choice[i, t, j] | choice_prob[i, t, j]);
      }

      if (t == Tsubj[i])
        continue;  // Skip the terminal state as there are no updates or choices

      row_vector[S] new_state = to_row_vector(state[i, t + 1]);

      for (j in 1:C) {
        if (choice[i, t, j] == 1) {
          PE = gamma^(week[i, t + 1] - week[i, t])
               * dot_product(w[j], new_state)
               - dot_product(w[j], current_state);
          delta = (outcome[i, t] - cost[j]) + PE;

          // Update w and theta for next round
          w[j] += alpha[1] * delta * to_vector(current_state);
          theta[j] += alpha[2] * delta * to_vector(current_state) * tau
                        ./ (1 + exp(dot_product(theta[j], current_state) * tau));
        }
      }
    }
  }
}
