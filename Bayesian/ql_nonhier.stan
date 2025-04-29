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
  // Individual parameters
  vector<lower=0, upper=1>[N] alpha;  // Learning rate
  vector<lower=0, upper=1>[N] gamma;  // Discount rate
  vector<lower=0>[N] tau;             // Inverse temperature
  vector[N] ev_init;                  // Initial expected value
  vector<lower=0>[N] cost;            // Cost
}

model {
  // Simple independent priors
  alpha ~ beta(2, 2);         // Centered at 0.5
  gamma ~ beta(2, 2);         // Centered at 0.5
  tau ~ normal(5, 3);         // Reasonable range for inverse temperature
  ev_init ~ normal(0, 1);     // Initial expected values
  cost ~ normal(0.5, 1);

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
