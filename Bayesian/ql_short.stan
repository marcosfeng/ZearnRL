data {
  int<lower=1> N;  // Number of subjects
  int<lower=1> T;  // Maximum number of trials
  array[N] int<lower=1, upper=T> Tsubj;  // Number of trials per subject
  array[N, T] int<lower=0, upper=1> choice;  // Choice for each trial
  array[N, T] real reward;  // Reward for each trial
  array[N, T] int<lower=0> week;  // Week number for each trial
}

parameters {
  // Population-level hyperparameters
  real<lower=0> a_alpha;  // Alpha beta distribution parameter a
  real<lower=0> b_alpha;  // Alpha beta distribution parameter b
  real<lower=0> a_gamma;  // Gamma beta distribution parameter a
  real<lower=0> b_gamma;  // Gamma beta distribution parameter b
  real mu_tau;    // Log-normal mu for tau
  real<lower=0> sigma_tau;  // Log-normal sigma for tau
  real mu_ev;     // Log-normal mu for initial expected value
  real<lower=0> sigma_ev;   // Log-normal sigma for initial expected value
  real mu_cost;   // Log-normal mu for cost
  real<lower=0> sigma_cost; // Log-normal sigma for cost

  // Individual-level parameters
  vector<lower=0, upper=1>[N] alpha;  // Learning rate (beta distributed)
  vector<lower=0, upper=1>[N] gamma;  // Discount rate (beta distributed)
  vector<lower=0>[N] tau;    // Inverse temperature (log-normal distributed)
  vector<lower=0>[N] ev;     // Initial expected value (log-normal distributed)
  vector<lower=0>[N] cost;   // Cost (log-normal distributed)
}

model {
  // Priors for population-level hyperparameters
  a_alpha ~ normal(2, 1);
  b_alpha ~ normal(2, 1);
  a_gamma ~ normal(2, 1);
  b_gamma ~ normal(2, 1);
  mu_tau ~ normal(0, 1);
  sigma_tau ~ normal(0, 0.5);
  mu_ev ~ normal(0, 1);
  sigma_ev ~ normal(0, 0.5);
  mu_cost ~ normal(0, 1);
  sigma_cost ~ normal(0, 0.5);

  // Individual-level parameters
  alpha ~ beta(a_alpha, b_alpha);
  gamma ~ beta(a_gamma, b_gamma);
  tau ~ lognormal(mu_tau, sigma_tau);
  ev ~ lognormal(mu_ev, sigma_ev);
  cost ~ lognormal(mu_cost, sigma_cost);

  // Loop through subjects
  for (i in 1:N) {
    vector[T] q_values;  // Q-values for each trial

    // Initialize Q-values
    q_values = rep_vector(ev[i], T);

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
    q_values = rep_vector(ev[i], T);

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
