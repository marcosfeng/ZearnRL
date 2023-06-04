//
// This Stan model defines a Q-learning model
// and applies it to the data.
//
data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1, upper=T> Tsubj[N]; // Rounds total for each subject
  int<lower=0> week[N,T]; // Number of the week
  int<lower=0, upper=1> choice[N, T, 4]; // Choice for each component (as binary values)
  real outcome[N, T];  // log badges
}
transformed data {
  vector[4] initV;  // initial values for EV
  initV = rep_vector(0.5, 4);
}
parameters {
  // Declare all parameters as vectors for vectorizing
  real<lower=0, upper=3> cost[3];        // cost in badge-units for each component
  real<lower=0, upper=1> gamma;      // discount rate
  real<lower=0, upper=1> alpha;    // step-sizes for each component
  real<lower=0> tau;               // inverse temperature
  real<lower=0> sensi[3];       // reward sensitivity for each of the four components
}
model {
  // pooled parameters
  for (j in 1:3) {
    cost[j]     ~ normal(1,2) T[0, 3];
    sensi[j]    ~ normal(1,2) T[0, ];
  }
  alpha         ~ uniform(0,1);
  tau           ~ normal(3,2) T[0, ];
  gamma         ~ uniform(0,1);

  // subject loop and trial loop
  for (i in 1:N) {
    vector[4] ev = initV; // expected value
    real PE[4];      // prediction error for each of the four components

    for (t in 1:(Tsubj[i] - 1)) {
      // compute action probabilities
      for (j in 1:4) {
        choice[i, t, j] ~ bernoulli_logit(tau * ev[j]);
      }
      // prediction error
      for (j in 1:3) {
        if (choice[i, t, j] == 1) {
          PE[j] = gamma^(week[i, t + 1] - week[i, t]) * sensi[j]
                * (outcome[i, t + 1] - cost[j])
                - ev[j];
          // value updating (learning)
          ev[j] += alpha * PE[j];
        }
      }
      if (choice[i, t, 4] == 1) {
        PE[4] = gamma^(week[i, t + 1] - week[i, t]) * outcome[i, t + 1]
                - ev[4];
        // value updating (learning)
        ev[4] += alpha * PE[4];
      }
    }
    //Last weeks:
    // compute action probabilities
    for (j in 1:4) {
      choice[i, Tsubj[i], j] ~ bernoulli_logit(tau * ev[j]);
    }
  }
}
