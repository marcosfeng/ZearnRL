function [loglik, prob, choice, q_values] = q_simple_posterior(parameters, subj)

    % Extract parameters
    nd_alpha = parameters(1);
    alpha = 1 / (1 + exp(-nd_alpha));
    nd_gamma = parameters(2);
    gamma = 1 / (1 + exp(-nd_gamma));
    nd_tau = parameters(3);
    tau = exp(nd_tau);

    % Unpack data
    Tsubj = length(subj.action);
    choice = subj.action;
    outcome = subj.outcome;
    week = subj.simmed.week;

    % Expected value (Q-value) difference initialized at 0
    q_values = zeros(1, Tsubj);

    % Save log probability of choice
    log_p = nan(Tsubj, 1);
    log_p(1) = choice(1) * (-log1p(exp(-tau * (q_values(:,1))))) + ...
        (1 - choice(1)) * (-log1p(exp(tau * q_values(:,1))));

    % Loop through trials
    for t = 2:Tsubj
        % Read info for the current trial
        a = choice(t);
        w_t = week(t);
        w_t_prev = week(t-1);

        if choice(t-1) == 1
            % Update Q-value if choice was made
            delta = gamma^(double(w_t) - double(w_t_prev)) * ...
                outcome(t) - q_values(:,t-1);
            q_values(:,t) = q_values(:,t-1) + (alpha * delta);
        elseif choice(t-1) == 0
            % Update Q-value relative to outside option
            delta = gamma^(double(w_t) - double(w_t_prev)) * outcome(t);
            q_values(:,t) = q_values(:,t-1) - (alpha * delta);
        end

        % Log probability of choice
        if tau*q_values(:,t) < -8
            log_p(t) = a*tau*q_values(:,t);
        elseif tau*q_values(:,t) > 8
            log_p(t) = (1 - a)*(-tau*q_values(:,t));
        else
            log_p(t) = a * (-log1p(exp(-tau*q_values(:,t)))) + ...
                (1 - a) * (-log1p(exp(tau*q_values(:,t))));
        end
    end

    % Log-likelihood is defined as the sum of log-probability of choice data
    loglik = sum(log_p, "omitmissing");
    prob = [exp(log_p).*(1-choice) + (1-exp(log_p)).*choice, ...
        exp(log_p).*choice + (1-exp(log_p)).*(1-choice)];
end