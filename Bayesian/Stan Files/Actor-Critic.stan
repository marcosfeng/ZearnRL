//
// This Stan model defines a reinforcement learning model
// using eligibility traces and applies it to the data.
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
  array[C] real<lower=0, upper=1> cost;  // cost in badge-units for each component
  real<lower=0, upper=1> gamma;  // discount rate
  vector<lower=0, upper=1>[2] alpha;  // step-sizes
}
model {
  // Flat-ish priors
  cost    ~ normal(0.5, C);
  gamma   ~ uniform(0, 1);
  alpha   ~ uniform(0, 1);

  // subject loop and trial loop
  for (i in 1:N) {
    // Save histories of w's and theta's
    matrix[S,C] w;      // State-Value weights
    matrix[S,C] theta;  // Policy weights
    real delta;
    real PE; // prediction error for each of the four components
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
      for (j in 1:C) {
        if (choice[i, t, j] == 1) {
          PE = gamma^(week[i, t + 1] - week[i, t])
               * dot_product(w[, j], new_state)
               - dot_product(w[, j], current_state);
          delta = (outcome[i, t] - cost[j]) - PE;
          // Update w:
          w[, j] += alpha[1] * delta * to_vector(current_state);
          // Update theta:
          theta[, j] += alpha[2] * delta * to_vector(current_state);
          // Update theta:
          theta[, j] += alpha[2] * delta * to_vector(current_state) * -tau
                        ./ (1 + exp(dot_product(theta[, j], current_state) * tau)) ;
        }
      }
    }
  }
}
