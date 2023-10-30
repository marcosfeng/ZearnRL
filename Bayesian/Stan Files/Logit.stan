data {
  int<lower=0> N;  // number of observations
  int<lower=0> C;  // number of classrooms
  matrix[N, 3] X;  // predictor matrix
  array[N, 3] int<lower=0, upper=1> y;  // response variables
  array[N]    int<lower=1, upper=C> classroom;  // classroom identifier
}
parameters {
  array[3] real alpha;
  matrix[3, 3] beta;  // coefficients for each outcome variable
}
model {
  for (j in 1:3) {
    y[:, j] ~ bernoulli_logit(alpha[j] + X * beta[, j]);
  }
}
generated quantities {
  matrix[N, 3] y_pred;
  // For log likelihood calculation
  vector[C] log_lik;  // log likelihood per classroom

  for (c in 1:C) {
    log_lik[c] = 0;  // initialize log likelihood for each classroom
  }

  for (n in 1:N) {
    for (j in 1:3) {
      y_pred[n, j] = inv_logit(alpha[j] + X[n,] * beta[, j]);
      log_lik[classroom[n]] += bernoulli_lpmf(y[n, j] | y_pred[n, j]);  // accumulate log likelihood by classroom
    }
  }
}
