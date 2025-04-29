data {
  int<lower=1> N;  // Number of subjects
  int<lower=1> L;  // Total number of trials across all subjects
  array[N] int<lower=1> Tsubj;  // Number of trials per subject
  array[L] int<lower=0, upper=1> choice;  // Choices concatenated across subjects
  array[L] real reward;  // Rewards concatenated across subjects
  array[L] int<lower=0> week;  // Weeks concatenated across subjects
  array[N + 1] int<lower=1> starts;  // Starting indices for each subject's data
  int<lower=0, upper=1> mixis;  // Mixture sampling flag
}

transformed data {
  // Pre-compute week differences for efficiency
  array[L] real week_diff;

  for (i in 1:N) {
    int start = starts[i];
    int end = starts[i + 1] - 1;

    // First trial has no difference
    week_diff[start] = 0;

    // Compute differences for remaining trials
    for (t in (start + 1):end) {
      week_diff[t] = week[t] - week[t - 1];
    }
  }
}

parameters {
  // Population-level parameters
  real mu_alpha_raw;  // Mean learning rate
  real mu_gamma_raw;  // Mean discount rate
  real mu_tau_raw;    // Mean inverse temperature
  real mu_ev_init;    // Mean initial expected value
  real mu_cost_raw;   // Mean cost

  real<lower=0> sigma_alpha;
  real<lower=0> sigma_gamma;
  real<lower=0> sigma_tau;
  real<lower=0> sigma_cost;
  real<lower=0> sigma_ev_init;

  // Individual-level parameters
  vector[N] alpha_raw;  // Learning rate per subject
  vector[N] gamma_raw;  // Discount rate per subject
  vector[N] tau_raw;    // Inverse temperature per subject
  vector[N] ev_init_raw;    // Initial Expected value per subject
  vector[N] cost_raw;   // Cost per subject
}

transformed parameters {
  // Transform parameters to appropriate scales
  vector<lower=0, upper=1>[N] alpha;  // Learning rate
  vector<lower=0, upper=1>[N] gamma;  // Discount rate
  vector<lower=0>[N] tau;             // Inverse temperature
  vector<lower=0>[N] cost;            // Cost
  vector[N] ev_init;                  // Initial expected value

  {
    // Non-centered parameterization for better sampling
    alpha = inv_logit(mu_alpha_raw + sigma_alpha * alpha_raw);
    gamma = inv_logit(mu_gamma_raw + sigma_gamma * gamma_raw);
    tau = exp(mu_tau_raw + sigma_tau * tau_raw);
    cost = exp(mu_cost_raw + sigma_cost * cost_raw);
    ev_init = mu_ev_init + sigma_ev_init * ev_init_raw;
  }
}

model {
  // Vectorized priors for population-level parameters
  {
    vector[5] mu_raw = [mu_alpha_raw, mu_gamma_raw, mu_tau_raw, mu_ev_init, mu_cost_raw]';
    mu_raw ~ normal(0, 2);
  }
  // Half-Cauchy priors for population SDs
  {
    vector[5] sigma = [sigma_alpha, sigma_gamma, sigma_tau, sigma_cost, sigma_ev_init]';
    sigma ~ cauchy(0, 2.5);
  }

  // Priors for individual-level parameters
  // (standard normal due to non-centered parameterization)
  alpha_raw ~ std_normal();
  gamma_raw ~ std_normal();
  tau_raw ~ std_normal();
  cost_raw ~ std_normal();
  ev_init_raw ~ std_normal();

  // Likelihood
  {
    vector[L] log_lik;
    for (i in 1:N) {
      int start = starts[i];
      int end = starts[i + 1] - 1;
      vector[end - start + 1] q_values = rep_vector(ev_init[i], end - start + 1);
      real tau_i = tau[i];
      real cost_i = cost[i];
      real alpha_i = alpha[i];
      real gamma_i = gamma[i];

      // First trial
      log_lik[start] = bernoulli_logit_lpmf(choice[start] | tau_i * (q_values[1] - cost_i));

      // Remaining trials
      for (t in (start + 1):end) {
        int t_idx = t - start + 1;  // Index within q_values
        real gamma_power = gamma_i^week_diff[t];
        real reward_discounted = gamma_power * reward[t];

        if (choice[t - 1] == 1) {
          real delta = reward_discounted - q_values[t_idx - 1];
          q_values[t_idx] = fma(alpha_i, delta, q_values[t_idx - 1]);
        } else {
          q_values[t_idx] = fma(-alpha_i, reward_discounted, q_values[t_idx - 1]);
        }

        log_lik[t] = bernoulli_logit_lpmf(choice[t] | tau_i * (q_values[t_idx] - cost_i));
      }
    }
    target += sum(log_lik);
    if (mixis) {
      target += log_sum_exp(-log_lik);
    }
  }
}

generated quantities {
  vector[L] log_lik;
  vector[L] y_pred;

  {
    for (i in 1:N) {
      int start = starts[i];
      int end = starts[i + 1] - 1;
      vector[end - start + 1] q_values = rep_vector(ev_init[i], end - start + 1);
      real tau_i = tau[i];
      real cost_i = cost[i];
      real alpha_i = alpha[i];
      real gamma_i = gamma[i];

      // First trial
      y_pred[start] = inv_logit(tau_i * (q_values[1] - cost_i));
      log_lik[start] = bernoulli_logit_lpmf(choice[start] | tau_i * (q_values[1] - cost_i));

      // Remaining trials
      for (t in (start + 1):end) {
        int t_idx = t - start + 1;
        real gamma_power = gamma_i^week_diff[t];
        real reward_discounted = gamma_power * reward[t];

        if (choice[t - 1] == 1) {
          real delta = reward_discounted - q_values[t_idx - 1];
          q_values[t_idx] = fma(alpha_i, delta, q_values[t_idx - 1]);
        } else {
          q_values[t_idx] = fma(-alpha_i, reward_discounted, q_values[t_idx - 1]);
        }

        y_pred[t] = inv_logit(tau_i * (q_values[t_idx] - cost_i));
        log_lik[t] = bernoulli_logit_lpmf(choice[t] | tau_i * (q_values[t_idx] - cost_i));
      }
    }
  }
}
