data {
  int<lower=0> N;  // number of observations
  int<lower=0> C;  // number of classrooms
  matrix[N, 3] X;  // predictor matrix
  array[N, 3] int<lower=0, upper=1> y;  // response variables
  array[N]    int<lower=1, upper=C> classroom;  // classroom identifier
}
transformed data {
  array[C] int counts = rep_array(0, C);  // initialize counts
  int T;
  for (n in 1:N) {
    counts[classroom[n]] += 1;  // increment count for the classroom of the nth observation
  }
  T = max(counts);  // find the maximum count
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
  array[C, T, 3] real y_pred = rep_array(-1, C, T, 3);
  // For log likelihood calculation
  vector[C] log_lik;  // log likelihood per classroom
  int k;
  array[C] int week = rep_array(0, C);

  for (c in 1:C) {
    log_lik[c] = 0;  // initialize log likelihood for each classroom
  }

  for (n in 1:N) {
    k = classroom[n];
    week[k] += 1;
    for (j in 1:3) {
      y_pred[k, week[k], j] = inv_logit(alpha[j] + X[n,] * beta[, j]);
      log_lik[k] += bernoulli_lpmf(y[n, j] | y_pred[k, week[k], j]);
    }
  }
}
