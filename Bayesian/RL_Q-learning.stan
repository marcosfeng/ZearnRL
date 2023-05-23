//
// This Stan model defines a Q-learning model
// and applies it to the data.
//
data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1, upper=T> Tsubj[N]; // Rounds total for each subject
  int<lower=1> week[N,T]; // Number of the week
  int<lower=-1, upper=2> choice[N, T]; // Effort variable (as integers)
  real outcome[N, T];  // log badges
}
transformed data {
  vector[2] initV;  // initial values for EV
  initV = rep_vector(0.5, 2);
}
parameters {
  // Declare all parameters as vectors for vectorizing
  real<lower=0> cost;        // cost in badge-units
  real<lower=0.5, upper=1> gamma;      // discount rate
  real<lower=0, upper=1> alpha;    // step-sizes
  real<lower=0> tau;      // inverse temperature
  real sensi;       // reward sensitivity
}
model {
  // pooled parameters
  alpha         ~ beta(2, 2);
  cost          ~ beta(1, 1);
  tau           ~ normal(3,2) T[0, ];
  gamma         ~ beta(12, 4);
  sensi         ~ normal(1,2);

  // subject loop and trial loop
  for (i in 1:N) {
    vector[2] ev; // expected value
    real PE;      // prediction error

    ev = initV;
    for (t in 1:(Tsubj[i] - 1)) {
      // compute action probabilities
      choice[i, t] ~ categorical_logit(tau * ev);
      // prediction error
      if (choice[i, t] == 0) {
        PE = gamma^(week[i, t + 1] - week[i, t]) * sensi * outcome[i, t + 1]
              - ev[choice[i, t]];
      } else if (choice[i, t] == 1) {
        PE = gamma^(week[i, t + 1] - week[i, t]) * sensi * outcome[i, t + 1]
              - cost
              - ev[choice[i, t]];
      }
      // value updating (learning)
      ???ev[choice[i, t]] += alpha * PE;
    }
    //Last weeks:
    // compute action probabilities
    choice[i, Tsubj[i]] ~ categorical_logit(tau * ev);
  }
}
