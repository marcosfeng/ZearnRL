//
// This Stan model defines a Q-learning model
// and applies it to the data.
//
data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1> number_teachers; // Number of unique teachers
  array[N] int<lower=1, upper=T> Tsubj; // Rounds total for each subject
  array[N] int<lower=1> group; // Teacher ID for each classroom
  array[N,T] int<lower=0> week; // Number of the week
  array[N, T, 4] int<lower=0, upper=1> choice; // Choice for each component (as binary values)
  array[N, T] real outcome;  // log badges
}
transformed data {
  vector[4] initV;  // initial values for EV
  initV = rep_vector(0.5, 4);
}
parameters {
  // Hyperparameters
  vector[3] mu_cost;  // Mean cost in badge-units for each component
  vector<lower=0>[3] sigma_cost; // Standard deviation of cost
  real mu_sensi; // Mean reward sensitivity
  real<lower=0> sigma_sensi; // Standard deviation of reward sensitivity
  real mu_alpha; // Mean step-sizes
  real<lower=0> sigma_alpha; // Standard deviation of step-sizes
  real mu_tau; // Mean inverse temperature
  real<lower=0> sigma_tau; // Standard deviation of inverse temperature
  real mu_gamma; // Mean discount rate
  real<lower=0> sigma_gamma; // Standard deviation of discount rate

  // Group-level parameters
  array[number_teachers, 3] real<lower=0, upper=1> cost;  // cost in badge-units for each component
  array[number_teachers] real<lower=0, upper=1> gamma;      // discount rate
  array[number_teachers] real<lower=0, upper=1> alpha;    // step-sizes for each component
  array[number_teachers] real<lower=0.001> tau;       // inverse temperature
  array[number_teachers] real<lower=0> sensi;    // reward sensitivity for each of the four components
}
model {
  // Hyperparameters priors
  mu_cost ~ normal(0, 2);
  sigma_cost ~ cauchy(0, 2.5);
  mu_sensi ~ normal(0, 2);
  sigma_sensi ~ cauchy(0, 2.5);
  mu_alpha ~ normal(0, 2);
  sigma_alpha ~ cauchy(0, 2.5);
  mu_tau ~ normal(0, 2);
  sigma_tau ~ cauchy(0, 2.5);
  mu_gamma ~ normal(0, 2);
  sigma_gamma ~ cauchy(0, 2.5);

  // Group-level parameters priors
  for (j in 1:3) {
    cost[:,j] ~ normal(mu_cost[j], sigma_cost[j]);
  }
  sensi ~ normal(mu_sensi, sigma_sensi);
  alpha ~ normal(mu_alpha, sigma_alpha);
  tau ~ normal(mu_tau, sigma_tau);
  gamma ~ normal(mu_gamma, sigma_gamma);

  // subject loop and trial loop
  for (i in 1:N) {
    vector[4] ev = initV; // expected value
    array[4] real PE;      // prediction error for each of the four components

    for (t in 1:(Tsubj[i] - 1)) {
      // compute action probabilities
      for (j in 1:4) {
        choice[i, t, j] ~ bernoulli_logit(tau[group[i]] * ev[j]);
      }
      // prediction error
      for (j in 1:3) {
        if (choice[i, t, j] == 1) {
          PE[j] = gamma[group[i]]^(week[i, t + 1] - week[i, t]) * sensi[group[i]]
                * (outcome[i, t + 1] - cost[group[i],j])
                - ev[j];
          // value updating (learning)
          ev[j] += alpha[group[i]] * PE[j];
        }
      }
      if (choice[i, t, 4] == 1) {
        PE[4] = gamma[group[i]]^(week[i, t + 1] - week[i, t]) * sensi[group[i]]
                * outcome[i, t + 1]
                - ev[4];
        // value updating (learning)
        ev[4] += alpha[group[i]] * PE[4];
      }
    }
    //Last weeks:
    // compute action probabilities
    for (j in 1:4) {
      choice[i, Tsubj[i], j] ~ bernoulli_logit(tau[group[i]] * ev[j]);
    }
  }
}
