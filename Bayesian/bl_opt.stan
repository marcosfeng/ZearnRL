data {
  int<lower=1> N;  // Number of subjects
  int<lower=1> L;  // Total number of trials across all subjects
  array[N] int<lower=1> Tsubj;  // Number of trials per subject
  array[L] int<lower=0, upper=1> choice;  // Choices concatenated across subjects
  array[N + 1] int<lower=1> starts;  // Starting indices for each subject's data
  int<lower=0, upper=1> mixis;  // Mixture sampling flag
}

parameters {
  // Population-level parameters
  real mu_beta;  // Population mean on raw scale
  real<lower=0> sigma_beta;  // Population standard deviation

  // Individual-level parameters (non-centered parameterization)
  vector[N] beta_raw;  // Individual intercepts on raw scale
}

transformed parameters {
  // Transform parameters to appropriate scales
  vector[N] beta;  // Individual intercepts
  beta = mu_beta + sigma_beta * beta_raw;
}
model {
  // Population-level priors
  mu_beta ~ normal(0, 2);
  sigma_beta ~ cauchy(0, 2.5);

  // Individual-level parameters
  beta_raw ~ std_normal();

  // Likelihood
  vector[L] log_lik;
  for (i in 1:N) {
    int start = starts[i];
    int end = starts[i + 1] - 1;
    for (t in start:end) {
      log_lik[t] = bernoulli_logit_lpmf(choice[t] | beta[i]);
    }
  }
  target += sum(log_lik);
  if (mixis) {
    target += log_sum_exp(-log_lik);
  }
}

generated quantities {
  vector[L] log_lik;
  vector[L] y_pred;

  {
    int pos = 1;
    for (i in 1:N) {
      int start = starts[i];
      int end = starts[i + 1] - 1;
      int n_trials = end - start + 1;

      // Compute predictions and log likelihood for actual trials
      vector[n_trials] logit_p = rep_vector(beta[i], n_trials);
      y_pred[start:end] = inv_logit(logit_p);

      for (t in start:end) {
        log_lik[t] = bernoulli_logit_lpmf(choice[t] | beta[i]);
      }
    }
  }
}
