data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1, upper=T> Tsubj[N];
  int<lower=-1, upper=2> choice[N, T];
  real outcome[N, T];  // no lower and upper bounds
}
transformed data {
  vector[2] initV;  // initial values for EV
  real Gamma;
  initV = rep_vector(0.0, 2);
  Gamma = 0.95; // discount rate
}
parameters {
// Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  vector[5] mu_pr;
  vector<lower=0>[5] sigma;

  // Subject-level raw parameters
  vector[N] A_pr;        // learning rate
  vector[N] tau_pr;      // inverse temperature
  vector[N] w1_pr;
  vector[N] w2_pr;
  vector[N] cost;
}
transformed parameters {
  // subject-level parameters
  vector<lower=0, upper=1>[N] A;
  vector<lower=0, upper=10>[N] tau;
  vector<lower=0, upper=1>[N] w1;
  vector<lower=0, upper=1>[N] w2;

  for (i in 1:N) {
    A[i]     = Phi_approx(mu_pr[1]  + sigma[1]  * A_pr[i]);
    tau[i]   = Phi_approx(mu_pr[2]  + sigma[2]  * tau_pr[i])  * 10;
    w1[i]    = Phi_approx(mu_pr[3]  + sigma[3]  * w1_pr[i]);
    w2[i]    = Phi_approx(mu_pr[4]  + sigma[4]  * w2_pr[i]) * (1 - w1[i]);
  }
}
model {
  // Hyperparameters
  mu_pr  ~ normal(0, 1);
  sigma ~ normal(0, 0.2);

  // individual parameters
  A_pr     ~ normal(0, 1);
  tau_pr   ~ normal(0, 1);
  w1_pr    ~ normal(0, 1);
  w2_pr    ~ normal(0, 1);
  for (i in 1:N) {
    cost[i]  ~ normal(mu_pr[5], sigma[5]);
  }

  // subject loop and trial loop
  for (i in 1:N) {
    vector[2] ev; // expected value
    real PE;      // prediction error

    ev = initV;
    for (t in 1:(Tsubj[i] - 2)) {
      // compute action probabilities
      choice[i, t] ~ categorical_logit(tau[i] * ev);
      // prediction error
      if (choice[i, t] == 1) {
        PE =  (1 - w2[i] - w1[i]) * Gamma^2 * outcome[i, t + 2] 
              + w2[i]             * Gamma   * outcome[i, t + 1] 
              + w1[i]                       * outcome[i, t]
              - ev[choice[i, t]];
      } else if (choice[i, t] == 2) {
        PE = ((1 - w2[i] - w1[i]) * Gamma^2 * outcome[i, t + 2] 
              + w2[i]             * Gamma   * outcome[i, t + 1]
              + w1[i]                       * outcome[i, t] - cost[i])
              - ev[choice[i, t]];
      }
      // value updating (learning)
      ev[choice[i, t]] += A[i] * PE;
    }
    /////
    //Last two weeks:
    // compute action probabilities
    choice[i, Tsubj[i]-1] ~ categorical_logit(tau[i] * ev);
    // prediction error
    if (choice[i, Tsubj[i] - 1] == 1) {
      PE =  w2[i] * Gamma * outcome[i, Tsubj[i]] 
            + w1[i]       * outcome[i, Tsubj[i] - 1]
            - ev[choice[i, Tsubj[i] - 1]];
    } else if (choice[i, Tsubj[i]-1] == 2) {
      PE = (w2[i] * Gamma * outcome[i, Tsubj[i]]
            + w1[i]       * outcome[i, Tsubj[i] - 1] - cost[i])
            - ev[choice[i, Tsubj[i] - 1]];
    }
    // value updating (learning)
    ev[choice[i, Tsubj[i] - 1]] += A[i] * PE;
    ///
    choice[i, Tsubj[i]] ~ categorical_logit(tau[i] * ev);
  }
}
generated quantities {
  // For group level parameters
  real<lower=0, upper=1>  mu_A;
  real<lower=0, upper=10> mu_tau;
  real<lower=0, upper=1>  mu_w1;
  real<lower=0, upper=1>  mu_w2;

  // For log likelihood calculation
  real log_lik[N];

  // For posterior predictive check
  real y_pred[N, T];

  // Set all posterior predictions to 0 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
    }
  }

  mu_A      = Phi_approx(mu_pr[1]);
  mu_tau    = Phi_approx(mu_pr[2]) * 10;
  mu_w1     = Phi_approx(mu_pr[3]);
  mu_w2     = Phi_approx(mu_pr[4]) * (1 - mu_w1);
  
  { // local section, this saves time and space
    for (i in 1:N) {
      vector[2] ev; // expected value
      real PE;      // prediction error

      // Initialize values
      ev = initV;

      log_lik[i] = 0;

      for (t in 1:(Tsubj[i]-2)) {
        // compute log likelihood of current trial
        log_lik[i] += categorical_logit_lpmf(choice[i, t] | tau[i] * ev);

        // generate posterior prediction for current trial
        y_pred[i, t] = softmax(tau[i] * ev)[1];
        // prediction error
        if (choice[i, t] == 1) {
          PE =  (1- w2[i] - w1[i]) * Gamma^2 * outcome[i, t + 2]
                + w2[i]            * Gamma   * outcome[i, t + 1] 
                + w1[i]                      * outcome[i, t]            
                - ev[choice[i, t]];
        } else if (choice[i, t] == 2) {
          PE = ((1- w2[i] - w1[i]) * Gamma^2 * outcome[i, t + 2] 
                + w2[i]            * Gamma   * outcome[i, t + 1]
                + w1[i]                      * outcome[i, t] - cost[i])
                - ev[choice[i, t]];
        }
        // value updating (learning)
        ev[choice[i, t]] += A[i] * PE;
      }
      //Last two weeks:
      // compute log likelihood of second to last trial
      log_lik[i] += categorical_logit_lpmf(choice[i, Tsubj[i] - 1] | tau[i] * ev);
      // generate posterior prediction for second to last trial
      y_pred[i, Tsubj[i] - 1] = softmax(tau[i] * ev)[1];
      // prediction error
      if (choice[i, Tsubj[i]-1] == 1) {
        PE =  w2[i] * Gamma * outcome[i, Tsubj[i]]
              + w1[i]       * outcome[i, Tsubj[i] - 1]
              - ev[choice[i, Tsubj[i] - 1]];
      } else if (choice[i, Tsubj[i]-1] == 2) {
        PE = (w2[i] * Gamma * outcome[i, Tsubj[i]]
              + w1[i]       * outcome[i, Tsubj[i] - 1] - cost[i])
              - ev[choice[i, Tsubj[i] - 1]];
      }
      // value updating (learning)
      ev[choice[i, Tsubj[i] - 1]] += A[i] * PE;
      // compute log likelihood of last trial
      log_lik[i] += categorical_logit_lpmf(choice[i, Tsubj[i]] | tau[i] * ev);
      // generate posterior prediction for last trial
      y_pred[i, Tsubj[i]] = softmax(tau[i] * ev)[1];
    }
  }
}
