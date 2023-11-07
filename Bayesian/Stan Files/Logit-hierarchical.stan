data {
  int<lower=0> N;  // number of observations
  int<lower=0> C;  // number of classrooms
  matrix[N, 3] X;  // predictor matrix
  array[N, 3] int<lower=0, upper=1> y;  // response variables
  array[N]    int<lower=1, upper=C> classroom;  // classroom identifier
  int<lower=1> number_teachers; // Number of unique teachers
  array[C] int<lower=1> group; // Teacher ID for each classroom
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
  array[3] real mu_alpha;           // group-level mean for alpha
  array[3] real<lower=0> sigma_alpha;   // group-level sd for alpha
  array[3] vector[3] mu_beta;         // group-level mean for beta
  array[3] vector<lower=0>[3] sigma_beta; // group-level sd for beta

  array[3, number_teachers] real alpha;   // classroom-specific alphas
  array[3, 3, number_teachers] real beta; // classroom-specific betas
}
model {
  int teacher;
  mu_alpha ~ normal(0, 1);
  sigma_alpha ~ cauchy(0, 2.5);

  for (i in 1:3) {
    mu_beta[i] ~ normal(0, 1);
    sigma_beta[i] ~ cauchy(0, 2.5);
  }
  for (c in 1:number_teachers) {
    alpha[, c] ~ normal(mu_alpha, sigma_alpha);
    for (j in 1:3) {
      beta[j, , c] ~ normal(mu_beta[j], sigma_beta[j]);
    }
  }

  for (n in 1:N) {
    teacher = group[classroom[n]];
    for (j in 1:3) {
      y[n, j] ~ bernoulli_logit(alpha[j, teacher]
                + dot_product(X[n,], to_vector(beta[j, , teacher])));
    }
  }
}
generated quantities {
  int teacher;
  array[C, T, 3] real y_pred = rep_array(-1, C, T, 3);
  // For log likelihood calculation
  vector[C] log_lik;  // log likelihood per classroom
  array[N, T, C] int y_sim;
  int k;
  array[C] int week = rep_array(0, C);

  for (c in 1:C) {
    log_lik[c] = 0;  // initialize log likelihood for each classroom
  }

  for (n in 1:N) {
    k = classroom[n];
    teacher = group[classroom[n]];
    week[k] += 1;
    for (j in 1:3) {
      y_pred[k, week[k], j] = inv_logit(alpha[j, teacher]
                              + dot_product(X[n,], to_vector(beta[j, , teacher])));
      log_lik[k] += bernoulli_lpmf(y[n, j] | y_pred[k, week[k], j]);
      y_sim[k, week[k], j] = bernoulli_logit_rng(alpha[j, teacher]
                              + dot_product(X[n,], to_vector(beta[j, , teacher])));
    }
  }
}
