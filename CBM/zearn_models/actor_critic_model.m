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
    Tsubj = length(subj.actions);
    choice = subj.actions;
    outcome = subj.outcome;
    state = [ones(Tsubj, 1), subj.state];
    week = subj.simmed.week;

    % Initialize
    C = size(choice, 2);  % Number of choices
    D = size(state, 2);  % Dimensionality of state space
    w = w_init * ones(D, C);  % Critic's state-action value estimates
    theta = theta_init * ones(D, C);  % Actor's policy parameters
    p = zeros(size(subj.actions, 1), 1);  % Log probabilities of choices

    % Loop through trials
    for t = 1:Tsubj
        % End the loop if the week is zero
        w_t = week(t);  % Week on this trial
        if week(t) == 0
            break;
        end
        s = state(t, :);  % Current state (now a vector)
        a = choice(t, :);  % Action on this trial (vector of 0s and 1s)
        o = outcome(t);  % Outcome on this trial

        % Actor: Compute policy (log probability of taking each action)
        % Calculate the product s * theta * tau
        product = s * theta * tau;
        % Define a threshold for 'large' theta
        if product < -8
            log_policy = -product;
            log_policy_complement = zeros(1,C);
        elseif product > 8
            log_policy = zeros(1,C);
            log_policy_complement = product;
        else
            log_policy = -log1p(exp(-product));
            log_policy_complement = -log1p(exp(product));
        end

        % Precompute logical indices
        idx_a1 = a == 1;
        idx_a0 = a == 0;
    
        % Compute log probability of the chosen actions
        p(t) = log_policy*idx_a1' + log_policy_complement*idx_a0';

        % Critic: Compute TD error (delta)
        if t < Tsubj
            w_t_next = week(t + 1); % Week on next trial
            s_next = state(t + 1, :);  % State on next trial
            PE = gamma^(double(w_t_next) - double(w_t)) * (s_next * w) - (s * w);
        else
            PE = 0;  % Terminal state
        end
        delta = (o - cost) - PE;

        % Update weights
        % Derivative of the logistic function: a/(1 + e^(a x))
        theta = theta + alpha_theta * (-tau * s') * (1 + exp(s * theta * tau)).^(-1) .* delta;
        w = w + alpha_w * s' * delta;
    end

    % Compute log-likelihood
    loglik = double(sum(p));
end
