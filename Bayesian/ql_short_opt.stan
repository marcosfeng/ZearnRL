data {
  int<lower=1> N;  // Number of subjects
  int<lower=1> T;  // Maximum number of trials
  array[N] int<lower=1, upper=T> Tsubj;  // Number of trials per subject
  array[N, T] int<lower=0, upper=1> choice;  // Choice for each trial
  array[N, T] real reward;  // Reward for each trial
  array[N, T] int<lower=0> week;  // Week number for each trial
}

transformed data {
  // Pre-compute week differences for efficiency
  array[N, T] real week_diff;
  int max_Tsubj = max(Tsubj);

  for (i in 1:N) {
    for (t in 2:Tsubj[i]) {
      week_diff[i, t] = week[i, t] - week[i, t-1];
    }
  }
}

parameters {
  // Population-level hyperparameters
  real<lower=0> a_alpha;
  real<lower=0> b_alpha;
  real<lower=0> a_gamma;
  real<lower=0> b_gamma;
  real mu_tau;
  real<lower=0> sigma_tau;
  real mu_ev;
  real<lower=0> sigma_ev;
  real mu_cost;
  real<lower=0> sigma_cost;

  // Individual-level parameters
  vector<lower=0, upper=1>[N] alpha;
  vector<lower=0, upper=1>[N] gamma;
  vector<lower=0>[N] tau;
  vector<lower=0>[N] ev;
  vector<lower=0>[N] cost;
}

transformed parameters {
  // Pre-compute common expressions
  vector[N] neg_alpha = -alpha;  // Used in the no-choice case
  array[N] vector[T] q_values;

  {
    for (i in 1:N) {
      // Initialize Q-values once
      q_values[i] = rep_vector(ev[i], T);

      // Update Q-values for each trial
      for (t in 2:Tsubj[i]) {
        real gamma_power = gamma[i]^week_diff[i, t];
        real reward_discounted = gamma_power * reward[i, t];

        if (choice[i, t-1] == 1) {
          real delta = reward_discounted - q_values[i, t-1];
          q_values[i, t] = fma(alpha[i], delta, q_values[i, t-1]);
        } else {
          q_values[i, t] = fma(neg_alpha[i], reward_discounted, q_values[i, t-1]);
        }
      }
    }
  }
}

model {
  // Priors
  // Population level priors
  a_alpha ~ normal(2, 1);
  b_alpha ~ normal(2, 1);
  a_gamma ~ normal(2, 1);
  b_gamma ~ normal(2, 1);

  // Group the location parameters
  {
    vector[3] mu = [mu_tau, mu_ev, mu_cost]';
    mu ~ normal(0, 1);
  }

  // Group the scale parameters
  {
    vector[3] sigma = [sigma_tau, sigma_ev, sigma_cost]';
    sigma ~ normal(0, 0.5);
  }

  // Individual-level parameters
  alpha ~ beta(a_alpha, b_alpha);
  gamma ~ beta(a_gamma, b_gamma);
  tau ~ lognormal(mu_tau, sigma_tau);
  ev ~ lognormal(mu_ev, sigma_ev);
  cost ~ lognormal(mu_cost, sigma_cost);

  // Likelihood
  {
    for (i in 1:N) {
      real tau_i = tau[i];
      real cost_i = cost[i];
      vector[T] scaled_values = tau_i * (q_values[i] - cost_i);

      // Vectorized first trial
      target += bernoulli_logit_lpmf(choice[i, 1] | scaled_values[1]);

      // Remaining trials
      for (t in 2:Tsubj[i]) {
        target += bernoulli_logit_lpmf(choice[i, t] | scaled_values[t]);
      }
    }
  }
}

generated quantities {
  array[N, T] real log_lik;
  array[N, T] real y_pred;

  {
    for (i in 1:N) {
      real tau_i = tau[i];
      real cost_i = cost[i];
      vector[T] scaled_values = tau_i * (q_values[i] - cost_i);

      // Initialize arrays for this subject
      for (t in 1:T) {
        if (t <= Tsubj[i]) {
          y_pred[i, t] = inv_logit(scaled_values[t]);
          log_lik[i, t] = bernoulli_logit_lpmf(choice[i, t] | scaled_values[t]);
        } else {
          y_pred[i, t] = -1;
          log_lik[i, t] = 0;
        }
      }
    }
  }
}
