//
// This Stan model defines a kernelized Q-learning model
// and applies it to the data.
//
data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1> S;  // Number of states
  int<lower=2> C;  // Number of choices
  int<lower=1> K;  // Number of lags in kernel
  array[N] int<lower=1, upper=T> Tsubj; // Rounds total for each subject
  array[N,T] int<lower=0> week; // Number of the week
  array[N, T, C] int<lower=0, upper=1> choice; // Choice for each component (as binary values)
  array[N, T] int<lower=1, upper=2> state;  // state for each time step
  array[N, T] real outcome;  // log badges
}
parameters {
  // Declare all parameters as vectors for vectorizing
  array[C] real<lower=0, upper=1> cost;  // cost in badge-units for each component
  real<lower=0, upper=1> gamma;      // discount rate
  real<lower=0, upper=1> alpha;    // step-sizes for each component
  real<lower=0.001, upper=10> tau;    // inverse temperature
  matrix[C, S] initV;
}
model {
  // pooled parameters
  for (j in 1:C) {
    cost[j]     ~ normal(1,2);
  }
  alpha         ~ uniform(0,1);
  tau           ~ gamma(5,5);
  gamma         ~ uniform(0,1);
  for (j in 1:S) {
    initV[:, j] ~ normal(0, 2);
  }

  // subject loop and trial loop
  for (i in 1:N) {
    matrix[C, S] ev = initV; // expected value

    for (t in 1:Tsubj[i]) {
      // compute action probabilities
      for (j in 1:C) {
        choice[i, t, j] ~ bernoulli_logit(tau * ev[j, state[i, t]]);
        if (t == Tsubj[i])  // Last week
          continue;
        // kernel reward
        real ker_reward = 0;
        real ker_norm   = 0;
        real PE = 0;      // prediction error
        for (t_past in 1:K) {
          if (t_past > t)
            continue;
          real jaccard_sim = 0; // Similarity for each state-action pair
          if (choice[i, t, j] == choice[i, t - (t_past - 1), j]) {
            jaccard_sim += 0.5;
            if (state[i, t] == state[i, t - (t_past - 1)]) {
              jaccard_sim += 0.5;
            }
            ker_reward += gamma^(1 + week[i, t] - week[i, t - (t_past - 1)])
                          * jaccard_sim
                          * outcome[i, t - (t_past - 1)];
            ker_norm   += jaccard_sim;
          }
        }
        if (ker_norm != 0) {
          ker_reward /= ker_norm;
          PE = (ker_reward - cost[j])
               - ev[j, state[i, t]];
          // value updating (learning)
          ev[j, state[i, t]] += alpha * PE;
        }
      }
    }
  }
}
generated quantities {
  // For posterior predictive check
  array[N, T, C] real y_pred;
  // Set all posterior predictions to 0 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      for (j in 1:C) {  // Added loop over choices
        y_pred[i, t, j] = -1;
      }
    }
  }
  // For log likelihood calculation
  vector[N] log_lik;

  // subject loop and trial loop
  for (i in 1:N) {
    matrix[C, S] ev = initV; // expected value
    log_lik[i] = 0; // initialize log likelihood for each subject
    for (t in 1:Tsubj[i]) {
      // compute action probabilities
      for (j in 1:C) {
        y_pred[i, t, j] = inv_logit(tau * ev[j, state[i, t]]);
        log_lik[i] += bernoulli_lpmf(choice[i, t, j] | y_pred[i, t, j]);
        if (t == Tsubj[i])  // Last week
          continue;

        // kernel reward
        real ker_reward = 0;
        real ker_norm   = 0;
        real PE = 0;      // prediction error
        for (t_past in 1:K) {
          if (t_past > t)
            continue;
          real jaccard_sim = 0; // Similarity for each state-action pair
          if (choice[i, t, j] == choice[i, t - (t_past - 1), j]) {
            jaccard_sim += 0.5;
            if (state[i, t] == state[i, t - (t_past - 1)]) {
              jaccard_sim += 0.5;
            }
            ker_reward += gamma^(1 + week[i, t] - week[i, t - (t_past - 1)])
                          * jaccard_sim
                          * outcome[i, t - (t_past - 1)];
            ker_norm   += jaccard_sim;
          }
        }
        if (ker_norm != 0) {
          ker_reward /= ker_norm;
          PE = (ker_reward - cost[j])
               - ev[j, state[i, t]];
          // value updating (learning)
          ev[j, state[i, t]] += alpha * PE;
        }
      }
    }
  }
}
