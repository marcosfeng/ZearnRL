function [loglik] = actor_critic_model(parameters, subj)
    % Extract parameters
    nd_alpha_w = parameters(1);  % Learning rate for w (critic)
    alpha_w = 1/(1+exp(-nd_alpha_w));
    nd_alpha_theta = parameters(2);  % Learning rate for theta (actor)
    alpha_theta = 1/(1+exp(-nd_alpha_theta));
    nd_gamma = parameters(3);  % Discount factor
    gamma = 1/(1+exp(-nd_gamma));
    nd_tau = parameters(4);  % Inverse temperature for softmax action selection
    tau = exp(nd_tau);
    theta_init = parameters(5);
    w_init = parameters(6);
    nd_cost = parameters(7:end);  % Cost for each action
    cost = exp(nd_cost);

    % Unpack data
    Tsubj = length(subj.action);
    choice = subj.action;
    outcome = subj.outcome;
    state = [ones(Tsubj, 1), subj.state];
    week = subj.simmed.week;

    % Initialize
    % C = size(choice, 2);  % Number of choices
    D = size(state, 2);  % Dimensionality of state space
    % w = w_init * ones(D, C);
    % theta = theta_init * ones(D, C);
    w = w_init * ones(D, 1);  % Critic's state-action value estimates
    theta = theta_init * ones(D, 1);  % Actor's policy parameters
    p = zeros(Tsubj, 1);  % Log probabilities of choices

    % Loop through trials
    for t = 1:Tsubj
        % End the loop if the week is zero
        w_t = week(t);  % Week on this trial
        if week(t) == 0
            break;
        end
        s = state(t, :);  % Current state (now a vector)
        a = choice(t);  % Action on this trial (vector of 0s and 1s)
        % a = choice(t, :);
        o = outcome(t);  % Outcome on this trial

        % Actor: Compute policy (log probability of taking each action)
        % Calculate the product s * theta * tau
        product = s * theta * tau;
        % Define a threshold for 'large' theta
        if product < -8
            p(t) = p(t) - product*a';
        elseif product > 8
            p(t) = p(t) + (1-product)*(1-a)';
        else
            p(t) = p(t) - ...
                log1p(exp(-product))*a' - ...
                log1p(exp(product))*(1-a)';
        end

        % Critic: Compute TD error (delta)
        if t < Tsubj
            w_t_next = week(t + 1); % Week on next trial
            s_next = state(t + 1, :);  % State on next trial
            PE = gamma^(double(w_t_next) - double(w_t)) * ...
                (s_next * w) - (s * w);
        else
            PE = 0;  % Terminal state
        end
        delta = (o - cost) + PE;

        % Update weights
        % Derivative of the Log of logistic function = a/(1 + e^(a x)
        theta = theta + alpha_theta * ...
            gamma^(double(w_t) - double(week(1))) * ...
            (tau * s') * (1 + exp(product)).^(-1) .* delta;
        w = w + alpha_w * s' * delta;
    end

    % Compute log-likelihood
    loglik = double(sum(p));
end
