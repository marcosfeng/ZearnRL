data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1> S; // Number of states
  int<lower=1, upper=T> Tsubj[N]; // Rounds total for each subject
  int choice[N, T]; // Effort in log minutes
  real outcome[N, T];  // log badges
  matrix[T * N, S] states;// Add states (tower alerts and lagged badges and their squares)
}
transformed data {
  row_vector[S] init;  // initial values for z and new_states 
  init = rep_row_vector(0.0, S);
}
parameters {
// Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  // real<lower=0> a[6];
  // real<lower=0> b[6];
  // real<lower=0> mu_w[S];
  // real<lower=0> sigma_w[S];
  // real mu_t[S];
  // real<lower=0> sigma_t[S];
  
  // Subject-level raw parameters
  real<lower=0, upper=0.2>            cost;        // cost in badge-units
  real<lower=0.7, upper=1>            gamma;       // discount rate
  vector<lower=0, upper=0.2>[2]       alpha;       // step-sizes
  vector<lower=0, upper=1>[2]       lambda;        // trace-decays
  // matrix[N, S]                    w_init;         // initial values for w
  // matrix[N, S]                    theta_init;     // initial values for theta
}
model {
  // Hyperparameters
  // mu_t ~ normal(0, 1);

  // individual parameters

  alpha[1]      ~ beta(2, 2);
  alpha[2]      ~ beta(3, 12);
  lambda[1]     ~ beta(2, 2);
  lambda[2]     ~ beta(15, 3);
  cost          ~ beta(2, 20);
  gamma         ~ beta(15, 3);
  // w_init[n]        ~ normal(mu_w, sigma_w);
  // theta_init[n]    ~ normal(mu_t, sigma_t);
  
  // subject loop and trial loop
  for (i in 1:N) {

    // Save histories of w's and theta's
    row_vector[S] w;
    row_vector[S] theta;
    row_vector[S] z_w;   // eligibility trace vector
    row_vector[S] z_t;   // eligibility trace vector
    real          delta;
    row_vector[S] new_states; // a place to store future states
    
    // w = w_init[i];
    // theta = theta_init[i];
    w = init;
    theta = init;
    
    z_w = init;
    z_t = init;
    // print("N: ", i);
    
    for (t in 1:Tsubj[i]) {
      //Find choice probability
      choice[i, t] ~ bernoulli_logit( dot_product(theta, states[(T * (i - 1) + t)]) );
      
      if (t != Tsubj[i])
        new_states = states[(T * (i - 1) + (t + 1))];
      else if (t == Tsubj[i])
        new_states = rep_row_vector(0.0, S);
      
      delta = outcome[i, t] - cost*choice[i, t]
              + gamma * dot_product(w, new_states)
              -         dot_product(w, states[(T * (i - 1) + t)]);
      
      z_w = gamma * lambda[1] * z_w 
            + states[(T * (i - 1) +  t)];
      z_t = gamma * lambda[2] * z_t 
            + states[(T * (i - 1) +  t)] 
              * gamma^(t-1) 
              * (1 - inv_logit(dot_product(theta, states[(T * (i - 1) + t)])) );
      
      // Update w:
      w     += alpha[1] * delta * z_w;
      // Update theta:
      theta += alpha[2] * delta * z_t;
    }
  }
}
