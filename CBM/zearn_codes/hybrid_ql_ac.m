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
    ev_init = parameters(4);
    nd_cost = parameters(5);
    cost = exp(nd_cost);

    % Initialize Q-value for each action
    C = length(cost);  % Number of choices
    ev = ev_init*ones(C);  % Expected value (Q-value)

    % To save log probability of choice.
    pred1 = zeros(Tsubj, 1);
    pred1(1) = 1 / (1 + exp(-tau * ev));
    
    % Loop through trials
    for t = 2:Tsubj
        % Read info for the current trial
        w_t = week(t);  % Week on this trial
        w_t_prev = week(t-1); % Week on next trial

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

        % Log probability of choice
        pred1(t) = 1 / (1 + exp(-tau * ev));
    end

    %% Actor-Critic
    state = [ones(Tsubj, 1), subj.state];
    D = size(state, 2);  % Dimensionality of state space

    nd_alpha_w = parameters(7);  % Learning rate for w (critic)
    alpha_w = 1/(1+exp(-nd_alpha_w));
    nd_alpha_theta = parameters(8);  % Learning rate for theta (actor)
    alpha_theta = 1/(1+exp(-nd_alpha_theta));
    nd_gamma = parameters(9);  % Discount factor
    gamma = 1/(1+exp(-nd_gamma));
    nd_tau = parameters(10);  % Inverse temperature for softmax action selection
    tau = exp(nd_tau);
    nd_cost = parameters(11);  % Cost for action = 1
    cost = exp(nd_cost);
    theta_init = parameters(12:(11+D));
    w_init = parameters((12+D):(11+2*D));

    % Initialize
    w = w_init';  % Critic's state-action value
    theta = theta_init';  % Actor's policy parameters
    pred2 = zeros(Tsubj, 1);  % Log probabilities of choices

    % Actor: Compute policy (log probability of taking each action)
    % Calculate the product s(t=1) * theta * tau
    product = state(1,:) * theta * tau;
    pred2(1) = 1 / (1 + exp(-product));

    % Loop through trials
    for t = 2:Tsubj
        % End the loop if the week is zero
        if week(t) == 0
            break;
        end
        week_t = week(t);  % Week on this trial
        week_t_1 = week(t - 1); % Week on previous trial
        s_t = state(t,:);  % Current state
        s_t_1 = state(t-1,:); % Previous state
        o = outcome(t);  % Outcome on this trial

        % Critic: Compute TD error (delta)
        if t < Tsubj
            PE = gamma^(double(week_t) - double(week_t_1)) * ...
                (s_t * w) - (s_t_1 * w);
        else
            PE = 0;  % Terminal state
        end
        delta = o - choice(t-1)*cost + PE;
        % Update weights
        % Derivative of the Log of logistic function = a/(1 + e^(a x))
        theta = theta + alpha_theta * ...
            gamma^(double(week_t_1) - double(week(1))) * ...
            (tau * s_t_1') * (1 + exp(product))^(-1) * delta;
        w = w + alpha_w * s_t_1' * delta;

        % Actor: Compute policy (log probability of taking each action)
        % Calculate the product s_t * theta * tau
        product = s_t * theta * tau;
        pred2(t) = 1 / (1 + exp(-product));
    end

    %% Hybrid
    nd_weight = parameters(6);
    weight = 1 / (1 + exp(-nd_weight));
    pred_hybrid = weight * pred1 + (1 - weight) * pred2;

    % Calculate the log likelihood
    loglik = double(choice)' * log(pred_hybrid) + ...
         double(1 - choice)' * log(1-pred_hybrid);
end
