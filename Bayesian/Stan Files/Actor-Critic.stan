//
// This Stan model defines an Actor-Critic RL model
//
data {
  int<lower=1> N;  // Number of subjects
  int<lower=1> T;  // Maximum number of rounds
  int<lower=1> S;  // Number of states
  array[N] int<lower=1, upper=T> Tsubj;  // Rounds total for each subject
  array[N,T] int<lower=0,upper=1> choice;  // Choice for each component (as binary values)
  array[N, T] real outcome;  // log badges
  array[N, T, S] real state;  // State Variables
  array[N,T] int<lower=0> week; // Number of the week
}
parameters {
  real<lower=0> cost;  // cost in badge-units for each component
  real<lower=0, upper=1> gamma;  // discount rate
  real<lower=0> tau;  // temperature
  vector<lower=0, upper=1>[2] alpha;  // step-sizes
  array[S] real w_0;  // initial Ws
  array[S] real theta_0;  // initial Thetas
}
model {
  // Priors from CBM analysis (top_model_comp[["cbm"]][[5]][[3]][[3]][[1]])
  cost     ~ normal(0.8, 0.3);
  gamma    ~ normal(0.8, 0.2);
  tau      ~ normal(7.5, 0.3);
  alpha    ~ normal(0.5, 0.2);
  w_0      ~ normal(0.2, 0.1);
  theta_0  ~ normal(0,   0.1);

  // subject loop and trial loop
  for (i in 1:N) {
    // Save histories of w's and theta's
    vector[S] w = to_vector(w_0);      // State-Value weights
    vector[S] theta = to_vector(theta_0);  // Policy weights
    real delta;
    real PE; // prediction error for each of the four components
    row_vector[S] current_state = rep_row_vector(0, S);

    for (t in 1:Tsubj[i]) {
      //Assign current state to a temporary variable
      if (t != 1) {
        current_state = to_row_vector(state[i, t - 1]);
      }
      // if  (t == 10) {
      //   print("theta: ", theta);
      //   print("current_state: ", current_state);
      //   print("tau: ", tau);
      // }
      //Find choice probability
      choice[i, t] ~ bernoulli_logit(tau * dot_product(theta, current_state));
      if (t == Tsubj[i])
        continue; // Terminal State

      //Assign next state to a temporary variable
      row_vector[S] new_state = to_row_vector(state[i, t]);
      // prediction error and update equations
      PE = gamma^(week[i, t + 1] - week[i, t])
           * dot_product(w, new_state)
           - dot_product(w, current_state);
      delta = (outcome[i, t] - cost) + PE;

      // Update w:
      w += alpha[1] * delta * to_vector(current_state);
      // Update theta:
      theta += alpha[2] * delta * to_vector(current_state) * tau * gamma^(week[i, t] - 1)
            ./ (1 + exp(dot_product(theta, current_state) * tau));
    }
  }
}
generated quantities {
  array[N, T] real choice_prob;  // Probability of choosing each component
  // For log likelihood calculation
  vector[N] log_lik;

  for (i in 1:N) {
    vector[S] w = to_vector(w_0);      // Initialize at w_0
    vector[S] theta = to_vector(theta_0);  // Initialize at theta_0
    real delta;
    real PE;  // Prediction error
    log_lik[i] = 0; // initialize log likelihood for each subject
    row_vector[S] current_state = rep_row_vector(0, S);

    for (t in 1:Tsubj[i]) {
      if (t != 1) {
        current_state = to_row_vector(state[i, t - 1]);
      }

      // Compute the choice probability
      choice_prob[i, t] = inv_logit( tau * dot_product(theta, current_state) );
      log_lik[i] += bernoulli_lpmf(choice[i, t] | choice_prob[i, t]);

      if (t == Tsubj[i])
        continue;  // Skip the terminal state as there are no updates or choices

      row_vector[S] new_state = to_row_vector(state[i, t]);
      PE = gamma^(week[i, t + 1] - week[i, t])
           * dot_product(w, new_state)
           - dot_product(w, current_state);
      delta = (outcome[i, t] - cost) + PE;

      // Update w and theta for next round
      w += alpha[1] * delta * to_vector(current_state);
      theta += alpha[2] * delta * to_vector(current_state) * tau * gamma^(week[i, t] - 1)
                    ./ (1 + exp(dot_product(theta, current_state) * tau));
    }
  }
}
