data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1> S;  // Number of states
  int<lower=1> G;  // Number of groups
  array[N] int<lower=1, upper=T> Tsubj; // Rounds total for each subject
  array[N] int<lower=1, upper=G> group; // Group for each subject
  array[N,T] int<lower=0> week; // Number of the week
  array[N, T] int<lower=0, upper=1> choice; // Choice for each component (as binary values)
  array[N, T, S] real state;  // state for each time step
  array[N, T] real outcome;  // log badges
}
parameters {
  // Group-level parameters
  real<lower=0> mu_cost;  // Mean cost in badge-units for each component
  real<lower=0.6, upper=1> mu_gamma;      // Mean discount rate
  vector<lower=0, upper=1>[2] mu_alpha;      // Mean step-size
  real<lower=0> mu_tau;             // Mean inverse temperature

  // Subject-level parameters
  vector[G] cost;  // cost in badge-units for each component per group
  vector[G] gamma;    // discount rate per group
  matrix[2, G] alpha;    // step-sizes per group
  vector[G] tau;      // inverse temperature per group
  array[G] vector[S] w_0;  // initial Ws
  array[G] vector[S] theta_0;  // initial Thetas
}
model {
  // Priors for group-level parameters
  // Values from CBM analysis (top_model_comp[["cbm"]][[5]][[3]][[3]][[1]])
  mu_cost     ~ normal(0.8, 0.6);
  mu_gamma    ~ normal(0.8, 0.4);
  mu_alpha    ~ normal(0.5, 0.4);
  mu_tau      ~ normal(7.5, 0.6);

  // Distributions for subject-level parameters
  for (g in 1:G) {
    cost[g]    ~ normal(mu_cost,  0.6);
    gamma[g]   ~ normal(mu_gamma, 0.4);
    alpha[:,g] ~ normal(mu_alpha, 0.4);
    tau[g]     ~ normal(mu_tau,   0.6);
    w_0[g]     ~ normal(0.2,      0.2);
    theta_0[g] ~ normal(0,        0.2);
  }

  // Subject loop and trial loop
  for (i in 1:N) {
    vector[S] w = w_0[group[i]];      // State-Value weights
    vector[S] theta = theta_0[group[i]];  // Policy weights
    real delta;
    real PE;
    row_vector[S] current_state = rep_row_vector(0, S);

    for (t in 1:Tsubj[i]) {
      //Assign current state to a temporary variable
      if (t != 1) {
        current_state = to_row_vector(state[i, t - 1]);
      }
      // compute action probabilities
      choice[i, t] ~ bernoulli_logit(tau[group[i]] * dot_product(theta, current_state));
      if (t == Tsubj[i])  // Terminal State
        continue;

      //Assign next state to a temporary variable
      row_vector[S] new_state = to_row_vector(state[i, t]);
      // prediction error and update equations
      PE = gamma[group[i]]^(week[i, t + 1] - week[i, t])
           * dot_product(w, new_state)
           - dot_product(w, current_state);
      delta = (outcome[i, t] - cost[group[i]]) + PE;
      // Update w:
      w += alpha[1, group[i]] * delta * to_vector(current_state);
      // Update theta:
      theta += alpha[2, group[i]] * delta * to_vector(current_state)
               * tau[group[i]] * gamma[group[i]]^(week[i, t] - 1)
               ./ (1 + exp(dot_product(theta, current_state) * tau[group[i]]));
    }
  }
}
generated quantities {
  // For posterior predictive check
  array[N, T] real y_pred = rep_array(-1, N, T);
  vector[N] log_lik;
  array[N, T] int y_sim;

  // subject loop and trial loop
  for (i in 1:N) {
    vector[S] w = w_0[group[i]];  // Initialize at w_0
    vector[S] theta = theta_0[group[i]];  // Initialize at theta_0
    real delta;
    real PE;  // Prediction error
    row_vector[S] current_state = rep_row_vector(0, S);
    log_lik[i] = 0; // initialize log likelihood for each subject

    for (t in 1:Tsubj[i]) {
      //Assign current state to a temporary variable
      if (t != 1) {
        current_state = to_row_vector(state[i, t - 1]);
      }
      // compute action probabilities
      y_pred[i, t] = inv_logit(tau[group[i]] * dot_product(theta, current_state));
      log_lik[i] += bernoulli_lpmf(choice[i, t] | y_pred[i, t]);

      if (t == Tsubj[i])  // Terminal State
        continue;

      row_vector[S] new_state = to_row_vector(state[i, t]);
      PE = gamma[group[i]]^(week[i, t + 1] - week[i, t])
           * dot_product(w, new_state)
           - dot_product(w, current_state);
      delta = (outcome[i, t] - cost[group[i]]) + PE;
      // Update w:
      w += alpha[1, group[i]] * delta * to_vector(current_state);
      // Update theta:
      theta += alpha[2, group[i]] * delta * to_vector(current_state)
               * tau[group[i]] * gamma[group[i]]^(week[i, t] - 1)
               ./ (1 + exp(dot_product(theta, current_state) * tau[group[i]]));
    }
  }
}
