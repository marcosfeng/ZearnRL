//
// This Stan model defines a reinforcement learning model
// using eligibility traces and applies it to the data.
//
data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1> S; // Number of states
  int<lower=1, upper=T> Tsubj[N]; // Rounds total for each subject
  //  matrix[N, T] choice; // Effort variable
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
  // Hyper(group)-parameters
  real mu_pr[6];
  real<lower=0> sigma[6];

  // Subject-level raw parameters
  real cost_pr[N];        // cost in badge-units
  real gamma_pr[N];       // discount rate
  vector[2] alpha_pr[N];  // step-sizes
}
transformed parameters {
  // subject-level parameters
  real<lower=0.1, upper=1> cost[N];        // cost in badge-units
  real<lower=0.7, upper=1> gamma[N];      // discount rate
  vector<lower=0, upper=0.3>[2] alpha[N];    // step-sizes

  for (i in 1:N) {
    cost[i] = Phi_approx(mu_pr[1]  + sigma[1]  * cost_pr[i]) * (1-0.1) + 0.1;
    gamma[i] = Phi_approx(mu_pr[2]  + sigma[2]  * gamma_pr[i]) * (1-0.7) + 0.7;
    alpha[i,1] = Phi_approx(mu_pr[3]  + sigma[3]  * alpha_pr[i,1]) * 0.3;
    alpha[i,2] = Phi_approx(mu_pr[4]  + sigma[4]  * alpha_pr[i,2]) * 0.3;
  }
}
model {
  // Hyperparameters
  mu_pr ~ normal(0, 0.05);
  sigma ~ normal(0, 0.1);

  // individual parameters
  cost_pr ~ normal(0, 1);
  alpha_pr[,1] ~ normal(0, 1);
  alpha_pr[,2] ~ normal(0, 1);

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
      //Find choice probability
      choice[i, t] ~ bernoulli_logit( dot_product(theta, states[(T * (i - 1) + t)]) );
      if (t != Tsubj[i])
        new_states = states[(T * (i - 1) + (t + 1))];
      else if (t == Tsubj[i])
        new_states = rep_row_vector(0.0, S);
        
      delta = outcome[i, t] - cost[i]*choice[i, t]
              + gamma[i] * dot_product(w, new_states)
              - dot_product(w, states[(T * (i - 1) + t)]);
      // Update w:
      w += alpha[i,1] * delta * states[(T * (i - 1) +  t)];
      // Update theta:
      theta += alpha[i,2] * delta * states[(T * (i - 1) +  t)];
    }
  }
}
