function [loglik] = hybrid_ql_ac(parameters, subj)
    % Unpack data
    Tsubj = length(subj.action);
    choice = subj.action;  % dummy
    outcome = subj.outcome;
    week = subj.simmed.week;

    %% Q-learning
    nd_alpha = parameters(1);
    alpha = 1 / (1 + exp(-nd_alpha));
    nd_gamma = parameters(2);
    gamma = 1 / (1 + exp(-nd_gamma));
    nd_tau = parameters(3);
    tau = exp(nd_tau);
    nd_cost = parameters(4);
    cost = exp(nd_cost);

    % Initialize Q-value for each action
    C = length(cost);  % Number of choices
    ev = zeros(C);

    % To save log probability of choice.
    pred1 = zeros(Tsubj, 1);
    pred1(1) = 1 / (1 + exp(-tau * ev));

    % Loop through trials
    for t = 2:Tsubj
        % Read info for the current trial
        w_t = week(t);  % Week on this trial
        w_t_prev = week(t-1); % Week on next trial

        % Log probability of choice
        pred1(t) = 1 / (1 + exp(-tau * ev));

        if choice(t-1) == 1
            % Update expected value (ev) if choice was made
            delta = gamma^(double(w_t) - double(w_t_prev)) * ...
                outcome(t) - cost - ev;
            ev = ev + (alpha * delta);
        elseif choice(t-1) == 0
            % Update expected value (ev) relative to outside option
            delta = gamma^(double(w_t) - double(w_t_prev)) * outcome(t);
            ev = ev - (alpha * delta);
        end
    end

    %% Actor-Critic
    state = [ones(Tsubj, 1), subj.state];

    nd_alpha_w = parameters(6);  % Learning rate for w (critic)
    alpha_w = 1/(1+exp(-nd_alpha_w));
    nd_alpha_theta = parameters(7);  % Learning rate for theta (actor)
    alpha_theta = 1/(1+exp(-nd_alpha_theta));
    nd_gamma = parameters(8);  % Discount factor
    gamma = 1/(1+exp(-nd_gamma));
    nd_tau = parameters(9);  % Inverse temperature for softmax action selection
    tau = exp(nd_tau);
    theta_init = parameters(10);
    w_init = parameters(11);
    nd_cost = parameters(12);  % Cost for each action
    cost = exp(nd_cost);

    % Initialize
    D = size(state, 2);  % Dimensionality of state space
    w = w_init * ones(D, 1);  % Critic's state-action value estimates
    theta = theta_init * ones(D, 1);  % Actor's policy parameters
    pred2 = zeros(Tsubj, 1);  % Log probabilities of choices

    % Loop through trials
    for t = 1:Tsubj
        % End the loop if the week is zero
        w_t = week(t);  % Week on this trial
        if week(t) == 0
            break;
        end
        s = state(t, :);  % Current state (now a vector)
        o = outcome(t);  % Outcome on this trial

        % Actor: Compute policy (log probability of taking each action)
        % Calculate the product s * theta * tau
        product = s * theta * tau;
        pred2(t) = 1 / (1 + exp(-product));

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

    %% Hybrid
    nd_weight = parameters(5);
    weight = 1 / (1 + exp(-nd_weight));
    pred_hybrid = weight * pred1 + (1 - weight) * pred2;

    % Calculate the log likelihood
    loglik = double(choice)' * log(pred_hybrid) + ...
         double(1 - choice)' * log(1-pred_hybrid);

end
