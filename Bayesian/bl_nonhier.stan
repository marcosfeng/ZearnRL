data {
  int<lower=1> N;  // Number of subjects
  int<lower=1> L;  // Total number of trials across all subjects
  array[N] int<lower=1> Tsubj;  // Number of trials per subject
  array[L] int<lower=0, upper=1> choice;  // Choices concatenated across subjects
  array[N + 1] int<lower=1> starts;  // Starting indices for each subject's data
  int<lower=0, upper=1> mixis;  // Mixture sampling flag
}

parameters {
  vector[N] beta;    // Individual intercepts
}

model {
  // Individual-level parameters
  beta ~ normal(0.5, 0.5);

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
