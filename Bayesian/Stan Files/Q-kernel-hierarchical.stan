//
// This Stan model defines a Q-learning model
// and applies it to the data.
//
data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1> S;  // Number of states
  int<lower=2> C;  // Number of choices
  int<lower=1> K;  // Number of lags in kernel
  int<lower=1> number_teachers; // Number of unique teachers
  array[N] int<lower=1, upper=T> Tsubj; // Rounds total for each subject
  array[N] int<lower=1> group; // Teacher ID for each classroom
  array[N,T] int<lower=0> week; // Number of the week
  array[N, T, C] int<lower=0, upper=1> choice; // Choice for each component (as binary values)
  array[N, T] int<lower=1, upper=2> state;  // state for each time step
  array[N, T] real outcome;  // log badges
}
parameters {
  vector<lower=0, upper=1>[C] mu_cost; // mean of the group-level cost
  vector<lower=0>[C] sigma_cost; // standard deviation of the group-level cost
  array[number_teachers, C] real<lower=0, upper=1> cost;  // cost in badge-units for each component

  real<lower=0, upper=1> mu_gamma; // mean of the group-level discount rate
  real<lower=0> sigma_gamma; // standard deviation of the group-level discount rate
  array[number_teachers] real<lower=0, upper=1> gamma; // discount rate

  real<lower=0, upper=1> mu_alpha; // mean of the group-level step size
  real<lower=0> sigma_alpha; // standard deviation of the group-level step size
  array[number_teachers] real<lower=0, upper=1> alpha; // step size for each component

  real<lower=0, upper=10> mu_tau; // mean of the group-level inverse temperature
  real<lower=0> sigma_tau; // standard deviation of the group-level inverse temperature
  array[number_teachers] real<lower=0.001, upper=10> tau; // inverse temperature

  matrix[C, S] initV;  // initial values for EV
}
model {
  mu_cost ~ normal(0.5, 1);
  sigma_cost ~ cauchy(0, 2.5);
  mu_alpha ~ normal(0.5, 1);
  sigma_alpha ~ cauchy(0, 2.5);
  mu_tau ~ normal(1, 1);
  sigma_tau ~ cauchy(0, 2.5);
  mu_gamma ~ normal(0.7, 1);
  sigma_gamma ~ cauchy(0, 2.5);
  for (j in 1:S) {
    initV[:, j] ~ normal(0, 2);
  }

  // pooled parameters
  for (j in 1:C) {
    cost[:,j] ~ normal(mu_cost[j], sigma_cost[j]);
  }
  alpha ~ normal(mu_alpha, sigma_alpha);
  tau ~ normal(mu_tau, sigma_tau);
  gamma ~ normal(mu_gamma, sigma_gamma);

  // subject loop and trial loop
  for (i in 1:N) {
    matrix[C, S] ev = initV; // expected value
    for (t in 1:Tsubj[i]) {
      // compute action probabilities
      for (j in 1:C) {
        choice[i, t, j] ~ bernoulli_logit(tau[group[i]] * ev[j, state[i, t]]);
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
            ker_reward += gamma[group[i]]^(1 + week[i, t] - week[i, t - (t_past - 1)])
                          * jaccard_sim
                          * outcome[i, t - (t_past - 1)];
            ker_norm   += jaccard_sim;
          }
        }
        if (ker_norm != 0) {
          ker_reward /= ker_norm;
          PE = (ker_reward - cost[group[i],j])
               - ev[j, state[i, t]];
          // value updating (learning)
          ev[j, state[i, t]] += alpha[group[i]] * PE;
        }
      }
    }
  }
}
generated quantities {
  // For posterior predictive check
  array[N, T, C] real y_pred = rep_array(0, N, T, C);
  vector[N] log_lik;

  // subject loop and trial loop
  for (i in 1:N) {
    matrix[C, S] ev = initV; // expected value
    log_lik[i] = 0; // initialize log likelihood for each subject
    for (t in 1:Tsubj[i]) {
      // compute action probabilities
      for (j in 1:C) {
        log_lik[i] += bernoulli_logit_lpmf(choice[i, t, j] | tau[group[i]] * ev[j, state[i, t]]);
        y_pred[i, t, j] = inv_logit(tau[group[i]] * ev[j, state[i, t]]);
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
            ker_reward += gamma[group[i]]^(1 + week[i, t] - week[i, t - (t_past - 1)])
                          * jaccard_sim
                          * outcome[i, t - (t_past - 1)];
            ker_norm   += jaccard_sim;
          }
        }
        if (ker_norm != 0) {
          ker_reward /= ker_norm;
          PE = (ker_reward - cost[group[i],j])
               - ev[j, state[i, t]];
          // value updating (learning)
          ev[j, state[i, t]] += alpha[group[i]] * PE;
        }
      }
    }
  }
}

