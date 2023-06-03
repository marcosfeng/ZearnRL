//
// This Stan model defines a reinforcement learning model
// using eligibility traces and applies it to the data.
//
data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1> S; // Number of states
  int<lower=1, upper=T> Tsubj[N]; // Rounds total for each subject
  int<lower=0, upper=1> choice[N, T]; // Effort variable (as integers)
  real outcome[N, T];  // log badges
  matrix[T * N, S] states; // State Variables
}
transformed data {
  row_vector[S] init;  // initial values for theta and new_states
  init = rep_row_vector(0, S);
}
parameters {
  // Declare all parameters as vectors for vectorizing
  real<lower=0, upper=1> cost;        // cost in badge-units
  real<lower=0.6, upper=0.99> gamma;      // discount rate
  vector<lower=0, upper=0.5>[2] alpha;    // step-sizes
}
model {
  // pooled parameters
  alpha[1]      ~ beta(4, 13);
  alpha[2]      ~ beta(4, 13);
  cost          ~ beta(1, 1);
  gamma         ~ uniform(0.6, 0.99);

  // subject loop and trial loop
  for (i in 1:N) {
    // Save histories of w's and theta's
    row_vector[S] w;
    row_vector[S] theta;
    real delta;
    row_vector[S] new_states; // a place to store future states

    w = init;
    theta = init;

    for (t in 1:Tsubj[i]) {
      //Assign current state to a temporary variable
      row_vector[S] current_state = states[(T * (i - 1) + t)];
      //Find choice probability
      choice[i, t] ~ bernoulli_logit( dot_product(theta, current_state) );
      if (t != Tsubj[i])
        new_states = states[(T * (i - 1) + (t + 1))];
      else if (t == Tsubj[i])
        new_states = rep_row_vector(0.0, S);

      delta = outcome[i, t] - cost*choice[i, t]
              + gamma * dot_product(w, new_states)
              - dot_product(w, current_state);
      // Update w:
      w += alpha[1] * delta * current_state;
      // Update theta:
      theta += alpha[2] * delta * current_state;
    }
  }
}
