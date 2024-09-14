function [loglik] = q_model_simple(parameters, subj)
    
    % Extract parameters
    nd_alpha = parameters(1);
    alpha = 1 / (1 + exp(-nd_alpha));
    nd_gamma = parameters(2);
    gamma = 1 / (1 + exp(-nd_gamma));
    nd_tau = parameters(3);
    tau = exp(nd_tau);
    nd_cost = parameters(4:end);
    cost = exp(nd_cost);
    
    % Unpack data
    Tsubj = length(subj.action);
    choice = subj.action;  % dummy
    outcome = subj.outcome;
    week = subj.simmed.week;

    % Initialize Q-value for each action
    C = length(cost);  % Number of choices
    ev = zeros(C);  % Expected value (Q-value) for both actions initialized at 0

    % Save log probability of choice
    log_p = zeros(Tsubj, 1);
    log_p(1) = log_p(1) + ...
        choice(1)*(-log1p(exp(-tau * (ev - cost)))) + ...
        (1 - choice(1))*(-log1p(exp(tau * ev)));

    % Loop through trials
    for t = 2:Tsubj
        % Read info for the current trial
        a = choice(t);  % Action on this trial
        w_t = week(t);  % Week on this trial
        w_t_prev = week(t-1); % Week on next trial

        if choice(t-1) == 1
            % Update expected value (ev) if choice was made
            delta = gamma^(double(w_t) - double(w_t_prev)) * ...
                outcome(t) - ev;
            ev = ev + (alpha * delta);
        elseif choice(t-1) == 0
            % Update expected value (ev) relative to outside option
            delta = gamma^(double(w_t) - double(w_t_prev)) * outcome(t);
            ev = ev - (alpha * delta);
        end

        % Log probability of choice
        if tau * (ev - cost) < -8
            log_p(t) = log_p(t) + a*tau*(ev - cost);
        elseif tau * ev > 8
            log_p(t) = log_p(t) + (1 - a)*(-tau * ev);
        else
            log_p(t) = log_p(t) + ...
                a*(-log1p(exp(-tau*(ev - cost)))) + ...
                (1 - a)*(-log1p(exp(tau * ev)));
        end
    end
    
    % Log-likelihood is defined as the sum of log-probability of choice data
    loglik = sum(log_p);
end
