data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1> S;  // Number of states
  int<lower=2> C;  // Number of choices
  int<lower=1> G;  // Number of groups
  array[N] int<lower=1, upper=T> Tsubj; // Rounds total for each subject
  array[N] int<lower=1, upper=G> group; // Group for each subject
  array[N,T] int<lower=0> week; // Number of the week
  array[N, T, C] int<lower=0, upper=1> choice; // Choice for each component (as binary values)
  array[N, T, S] real state;  // state for each time step
  array[N, T] real outcome;  // log badges
}
parameters {
  // Group-level parameters
  vector<lower=0, upper=1>[C] mu_cost;  // Mean cost in badge-units for each component
  vector<lower=0>[C] sigma_cost;  // Standard deviation of cost for each component
  real<lower=0, upper=1> mu_gamma;      // Mean discount rate
  real<lower=0> sigma_gamma;            // Standard deviation of discount rate
  vector<lower=0, upper=1>[2] mu_alpha;      // Mean step-size
  vector<lower=0>[2] sigma_alpha;            // Standard deviation of step-size
  real<lower=0.001> mu_tau;             // Mean inverse temperature
  real<lower=0> sigma_tau;              // Standard deviation of inverse temperature
  array[C] vector[S] w_0;  // initial Ws
  array[C] vector[S] theta_0;  // initial Thetas

  // Subject-level parameters
  matrix[C, G] cost;  // cost in badge-units for each component per group
  vector[G] gamma;    // discount rate per group
  matrix[2, G] alpha;    // step-sizes per group
  vector[G] tau;      // inverse temperature per group
}
model {
  // Priors for group-level parameters
  mu_cost ~ exponential(3);
  sigma_cost ~ cauchy(0, 2.5);
  mu_gamma ~ uniform(0, 1);
  sigma_gamma ~ cauchy(0, 2.5);
  mu_alpha ~ uniform(0, 1);
  sigma_alpha ~ cauchy(0, 2.5);
  mu_tau ~ exponential(1.0/17);
  sigma_tau ~ cauchy(0, 2.5);

  for (c in 1:C) {
    w_0[c]      ~ normal(-1, 2);
    theta_0[c]  ~ normal(0, 2);
  }

  // Distributions for subject-level parameters
  for (g in 1:G) {
    cost[:,g] ~ normal(mu_cost, sigma_cost);
    gamma[g] ~ normal(mu_gamma, sigma_gamma);
    alpha[:,g] ~ normal(mu_alpha, sigma_alpha);
    tau[g] ~ normal(mu_tau, sigma_tau);
  }

  // Subject loop and trial loop
  for (i in 1:N) {
    array[C] vector[S] w = w_0;      // State-Value weights
    array[C] vector[S] theta = theta_0;  // Policy weights
    real delta;
    real PE;

    for (t in 1:Tsubj[i]) {
      //Assign current state to a temporary variable
      row_vector[S] current_state = to_row_vector(state[i, t]);
      // compute action probabilities
      for (j in 1:C) {
        choice[i, t, j] ~ bernoulli_logit(tau[group[i]] * dot_product(theta[j], current_state));
      }

      if (t == Tsubj[i])  // Last week
        continue;

      //Assign next state to a temporary variable
      row_vector[S] new_state = to_row_vector(state[i, t + 1]);
      // prediction error and update equations
      for (j in 1:C) {
        if (choice[i, t, j] == 1) {
          PE = gamma[group[i]]^(week[i, t + 1] - week[i, t])
               * dot_product(w[j], new_state)
               - dot_product(w[j], current_state);
          delta = (outcome[i, t] - cost[j, group[i]]) + PE;
          // Update w:
          w[j] += alpha[1, group[i]] * delta * to_vector(current_state);
          // Update theta:
          theta[j] += alpha[2, group[i]] * delta * to_vector(current_state) * tau[group[i]]
                        ./ (1 + exp(dot_product(theta[j], current_state) * tau[group[i]]));
        }
      }
    }
  }
}
generated quantities {
  // For posterior predictive check
  array[N, T, C] real y_pred = rep_array(-1, N, T, C);
  vector[N] log_lik;
  array[N, T, C] int y_sim;

  // subject loop and trial loop
  for (i in 1:N) {
    array[C] vector[S] w = w_0;  // Initialize at w_0
    array[C] vector[S] theta = theta_0;  // Initialize at theta_0
    real delta;
    real PE;  // Prediction error
    log_lik[i] = 0; // initialize log likelihood for each subject

    for (t in 1:Tsubj[i]) {
      row_vector[S] current_state = to_row_vector(state[i, t]);
      // compute action probabilities
      for (j in 1:C) {
        y_pred[i, t, j] = inv_logit(tau[group[i]] * dot_product(theta[j], current_state));
        log_lik[i] += bernoulli_lpmf(choice[i, t, j] | y_pred[i, t, j]);
        y_sim[i, t, j] = bernoulli_logit_rng(tau[group[i]] * dot_product(theta[j], current_state));
      }

      if (t == Tsubj[i])  // Last week
        continue;
      row_vector[S] new_state = to_row_vector(state[i, t + 1]);

      for (j in 1:C) {
        if (choice[i, t, j] == 1) {
          PE = gamma[group[i]]^(week[i, t + 1] - week[i, t])
               * dot_product(w[j], new_state)
               - dot_product(w[j], current_state);
          delta = (outcome[i, t] - cost[j, group[i]]) + PE;
          // Update w:
          w[j] += alpha[1, group[i]] * delta * to_vector(current_state);
          // Update theta:
          theta[j] += alpha[2, group[i]] * delta * to_vector(current_state) * tau[group[i]]
                        ./ (1 + exp(dot_product(theta[j], current_state) * tau[group[i]]));
        }
      }
    }
  }
}
