function [loglik, prob, choice, q_values] = q_posterior(parameters, subj)
    % Extract parameters
    nd_alpha = parameters(1);
    alpha = 1 / (1 + exp(-nd_alpha));
    nd_gamma = parameters(2);
    gamma = 1 / (1 + exp(-nd_gamma));
    nd_tau = parameters(3);
    tau = exp(nd_tau);
    ev_init = parameters(4);
    nd_cost = parameters(5:end);
    cost = exp(nd_cost);
    
    % Unpack data
    Tsubj = length(subj.action);
    choice = subj.action;  % dummy
    outcome = subj.outcome;
    week = subj.simmed.week;
    
    % Initialize Q-value for each action
    C = length(cost);  % Number of choices
    q_values = ev_init * ones(C, Tsubj); % Q-values for each action and trial
    
    % Save log probability of choice
    log_p = nan(Tsubj, 1);
    log_p(1) = choice(1) * (-log1p(exp(-tau * q_values(:,1)))) + ...
        (1 - choice(1)) * (-log1p(exp(tau * q_values(:,1))));
    
    % Loop through trials
    for t = 2:Tsubj
        % Read info for the current trial
        a = choice(t); % Action on this trial
        w_t = week(t); % Week on this trial
        w_t_prev = week(t-1); % Week on next trial
    
        if choice(t-1) == 1
            % Update Q-value if choice was made
            delta = gamma^(double(w_t) - double(w_t_prev)) * ...
                outcome(t) - cost - q_values(:,t-1);
            q_values(:,t) = q_values(:,t-1) + (alpha * delta);
        elseif choice(t-1) == 0
            % Update Q-value relative to outside option
            delta = gamma^(double(w_t) - double(w_t_prev)) * outcome(t);
            q_values(:,t) = q_values(:,t-1) - (alpha * delta);
        end
    
        % Log probability of choice
        logit_val = tau * q_values(:,t);
        if logit_val < -8
            log_p(t) = a * logit_val;
        elseif logit_val > 8
            log_p(t) = (1 - a) * (-logit_val);
        else
            log_p(t) = a * (-log1p(exp(-logit_val))) + ...
                (1 - a) * (-log1p(exp(logit_val)));
        end
    end
    
    % Log-likelihood is defined as the sum of log-probability of choice data
    loglik = sum(log_p,"omitmissing");
    prob = exp(log_p) .* choice + (1 - exp(log_p)) .* (1 - choice);
    prob = [1-prob, prob];
end