//
// This Stan model defines a Q-learning model
// and applies it to the data.
//
data {
  int<lower=1> N;
  int<lower=1> T;
  array[N] int<lower=1, upper=T> Tsubj; // Rounds total for each subject
  array[N,T] int<lower=0> week; // Number of the week
  array[N, T, 4] int<lower=0, upper=1> choice; // Choice for each component (as binary values)
  array[N, T] int<lower=1, upper=2> state;  // state for each time step
  array[N, T] real outcome;  // log badges
}
transformed data {
  matrix[4, 2] initV;  // initial values for EV
  for (j in 1:2) {
    initV[:, j] = rep_vector(0.5, 4);
  }
}
parameters {
  // Declare all parameters as vectors for vectorizing
  array[3] real<lower=0, upper=1> cost;  // cost in badge-units for each component
  real<lower=0, upper=1> gamma;      // discount rate
  real<lower=0, upper=1> alpha;    // step-sizes for each component
  real<lower=0.001> tau;               // inverse temperature
  real<lower=0> sensi;    // reward sensitivity for each of the four components
}
model {
  // pooled parameters
  for (j in 1:3) {
    cost[j]     ~ normal(1,2) T[0, ];

  }
  sensi         ~ normal(1,2) T[0, ];
  alpha         ~ uniform(0,1);
  tau           ~ gamma(5,5);
  gamma         ~ uniform(0,1);

  // subject loop and trial loop
  for (i in 1:N) {
    matrix[4, 2] ev = initV; // expected value
    array[4] real PE;      // prediction error for each of the four components

    for (t in 1:(Tsubj[i] - 1)) {
      // compute action probabilities
      for (j in 1:4) {
        choice[i, t, j] ~ bernoulli_logit(tau * ev[j, state[i, t]]);
      }
      // prediction error
      for (j in 1:3) {
        if (choice[i, t, j] == 1) {
          PE[j] = gamma^(week[i, t + 1] - week[i, t]) * sensi
                * (outcome[i, t + 1] - cost[j])
                - ev[j, state[i, t]];
          // value updating (learning)
          ev[j] += alpha * PE[j];
        }
      }
      if (choice[i, t, 4] == 1) {
        PE[4] = gamma^(week[i, t + 1] - week[i, t]) * sensi
                * outcome[i, t + 1]
                - ev[4, state[i, t]];
        // value updating (learning)
        ev[4] += alpha * PE[4];
      }
    }
    //Last weeks:
    // compute action probabilities
    for (j in 1:4) {
      choice[i, Tsubj[i], j] ~ bernoulli_logit(tau * ev[j, state[i, Tsubj[i]]]);
    }
  }
}
