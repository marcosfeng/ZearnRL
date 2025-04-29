data {
  int<lower=1> N;  // Number of subjects
  int<lower=1> T;  // Maximum number of trials
  array[N] int<lower=1, upper=T> Tsubj;  // Number of trials per subject
  array[N, T] int<lower=0, upper=1> choice;  // Choice for each trial (only choice 2)
  array[N, T] real reward;  // Reward for each trial (only reward 1)
  array[N, T] int<lower=0> week;  // Week number for each trial
}

parameters {
  // Population-level parameters
  real mu_alpha_raw;  // Mean learning rate
  real mu_gamma_raw;  // Mean discount rate
  real mu_tau_raw;    // Mean inverse temperature
  real mu_ev_init;    // Mean initial expected value
  real mu_cost_raw;   // Mean cost

  // Individual-level parameters
  vector[N] alpha_raw;   // Learning rate per subject
  vector[N] gamma_raw;   // Discount rate per subject
  vector[N] tau_raw;     // Inverse temperature per subject
  vector[N] ev_init_raw; // Initial expected value per subject
  vector[N] cost_raw;    // Cost per subject
}

transformed parameters {
  // Transform parameters to appropriate scales
  vector<lower=0, upper=1>[N] alpha;  // Learning rate
  vector<lower=0, upper=1>[N] gamma;  // Discount rate
  vector<lower=0>[N] tau;             // Inverse temperature
  vector<lower=0>[N] ev_init;         // Initial expected value
  vector<lower=0>[N] cost;            // Cost

  // Parameter transformations
  for (i in 1:N) {
    alpha[i] = 1.0 / (1.0 + exp(-alpha_raw[i]));
    gamma[i] = 1.0 / (1.0 + exp(-gamma_raw[i]));
    tau[i] = exp(tau_raw[i]);
    ev_init[i] = exp(ev_init_raw[i]);
    cost[i] = exp(cost_raw[i]);
  }
}

model {
  // Priors for population-level parameters
  mu_alpha_raw ~ normal(0, 1);
  mu_gamma_raw ~ normal(0, 1);
  mu_tau_raw ~ normal(0, 1);
  mu_ev_init ~ normal(0, 1);
  mu_cost_raw ~ normal(0, 1);

  // Individual-level parameters
  alpha_raw ~ normal(mu_alpha_raw, 0.5);
  gamma_raw ~ normal(mu_gamma_raw, 0.5);
  tau_raw ~ normal(mu_tau_raw, 0.5);
  ev_init_raw ~ normal(mu_ev_init, 0.5);
  cost_raw ~ normal(mu_cost_raw, 0.5);

  // Loop through subjects
  for (i in 1:N) {
    vector[T] q_values;  // Q-values for each trial

    // Initialize Q-values
    q_values = rep_vector(ev_init[i], T);

    // First trial
    choice[i, 1] ~ bernoulli_logit(tau[i] * (q_values[1] - cost[i]));

    // Loop through trials
    for (t in 2:Tsubj[i]) {
      real delta;

      // Update Q-values based on previous choice
      if (choice[i, t-1] == 1) {
        // Update if choice was made
        delta = gamma[i]^(week[i, t] - week[i, t-1]) *
                reward[i, t] - q_values[t-1];
        q_values[t] = q_values[t-1] + (alpha[i] * delta);
      } else {
        // Update relative to outside option
        delta = gamma[i]^(week[i, t] - week[i, t-1]) *
                reward[i, t];
        q_values[t] = q_values[t-1] - (alpha[i] * delta);
      }

      // Choice probability
      choice[i, t] ~ bernoulli_logit(tau[i] * (q_values[t] - cost[i]));
    }
  }
}

generated quantities {
  // For posterior predictive checks
  array[N, T] real log_lik;
  array[N, T] real y_pred;

  // Initialize arrays
  for (i in 1:N) {
    for (t in 1:T) {
      log_lik[i, t] = 0;
      y_pred[i, t] = -1;
    }
  }

  // Loop through subjects
  for (i in 1:N) {
    vector[T] q_values;

    // Initialize Q-values
    q_values = rep_vector(ev_init[i], T);

    // First trial
    y_pred[i, 1] = inv_logit(tau[i] * (q_values[1] - cost[i]));
    log_lik[i, 1] = bernoulli_logit_lpmf(choice[i, 1] |
                    tau[i] * (q_values[1] - cost[i]));

    // Loop through trials
    for (t in 2:Tsubj[i]) {
      real delta;

      // Update Q-values based on previous choice
      if (choice[i, t-1] == 1) {
        delta = gamma[i]^(week[i, t] - week[i, t-1]) *
                reward[i, t] - q_values[t-1];
        q_values[t] = q_values[t-1] + (alpha[i] * delta);
      } else {
        delta = gamma[i]^(week[i, t] - week[i, t-1]) *
                reward[i, t];
        q_values[t] = q_values[t-1] - (alpha[i] * delta);
      }

      // Generate predictions and log likelihood
      y_pred[i, t] = inv_logit(tau[i] * (q_values[t] - cost[i]));
      log_lik[i, t] = bernoulli_logit_lpmf(choice[i, t] |
                      tau[i] * (q_values[t] - cost[i]));
    }
  }
}
