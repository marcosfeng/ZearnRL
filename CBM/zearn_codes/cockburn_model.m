function [loglik] = cockburn_model(parameters, subj)
    % Extract parameters
    beta_0 = parameters(1);
    beta_ev = parameters(2);
    beta_sd = parameters(3);
    beta_t = parameters(4);
    beta_ev_t = parameters(5);
    beta_sd_t = parameters(6);
    
    % Unpack data
    choice = subj.action;
    ev = subj.ev;
    sd = subj.sd;
    week = double(subj.simmed.week);
    
    % Create unit-scaled trial number t
    t = (week - min(week)) / (max(week) - min(week));
    
    % Create lagged predictors
    lag_ev = [0; ev(1:end-1)];
    lag_sd = [0; fillmissing(sd(1:end-1), 'constant', 0)];
    
    % Compute linear predictor
    linear_pred = beta_0 + ...
                  beta_ev * lag_ev + ...
                  beta_sd * lag_sd + ...
                  beta_t * t + ...
                  beta_ev_t * lag_ev .* t + ...
                  beta_sd_t * lag_sd .* t;
    
    % Compute log probability of choice
    log_p = choice .* (-log1p(exp(-linear_pred))) + ...
            (1 - choice) .* (-log1p(exp(linear_pred)));
    
    % Compute log-likelihood
    loglik = sum(log_p,"omitnan");
end