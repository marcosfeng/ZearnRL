//
// This Stan model defines a state-free Q-learning model
// and applies it to the data.
//
data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=2> C;  // Number of choices
  array[N] int<lower=1, upper=T> Tsubj; // Rounds total for each subject
  array[N,T] int<lower=0> week; // Number of the week
  array[N, T, C] int<lower=0, upper=1> choice; // Choice for each component (as binary values)
  array[N, T] real outcome;  // log badges
}
parameters {
  // Declare all parameters as vectors for vectorizing
  array[C] real<lower=0, upper=1> cost; // cost in badge-units for each component
  real<lower=0, upper=1> gamma;        // discount rate
  real<lower=0, upper=1> alpha;       // step-sizes for each component
  real<lower=0.001> tau;             // inverse temperature
  vector[C] initV; // cost in badge-units for each component
}
model {
  // Priors
  for (j in 1:3) {
    cost[j]     ~ normal(1,2);
  }
  alpha         ~ uniform(0,1);
  tau           ~ gamma(5,5);
  gamma         ~ uniform(0,1);
  initV         ~ normal(0,2);

  // subject loop and trial loop
  for (i in 1:N) {
    vector[C] ev = initV; // expected value
    for (t in 1:Tsubj[i]) {
      // compute action probabilities
      for (j in 1:C) {
        choice[i, t, j] ~ bernoulli_logit(tau * ev[j]);
        if (t == Tsubj[i])  // Last week
          continue;
        real PE = 0;      // prediction error
        if (choice[i, t, j] == 1) {
          PE = gamma * (outcome[i, t] - cost[j]) - ev[j];
          // value updating (learning)
          ev[j] += alpha * PE;
        }
      }
    }
  }
}
generated quantities {
  // For posterior predictive check
  array[N, T, C] real y_pred;
  // Set all posterior predictions to 0 (avoids NULL values)
  y_pred = -1;
  // For log likelihood calculation
  vector[N] log_lik;

  // subject loop and trial loop
  for (i in 1:N) {
    vector[C] ev = initV; // expected value
    log_lik[i] = 0; // initialize log likelihood for each subject
    for (t in 1:Tsubj[i]) {
      // compute action probabilities
      for (j in 1:C) {
        y_pred[i, t, j] = inv_logit(tau * ev[j]);
        log_lik[i] += bernoulli_lpmf(choice[i, t, j] | y_pred[i, t, j]);
        if (t == Tsubj[i])  // Last week
          continue;
        real PE = 0;      // prediction error
        if (choice[i, t, j] == 1) {
          PE = gamma * (outcome[i, t] - cost[j]) - ev[j];
          // value updating (learning)
          ev[j] += alpha * PE;
        }
      }
    }
  }
}
