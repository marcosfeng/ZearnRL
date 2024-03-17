function [loglik] = hybrid_ac_logit(parameters, subj)
    % Unpack data
    Tsubj = length(subj.action);
    choice = subj.action;  % dummy
    outcome = subj.outcome;
    week = subj.simmed.week;

    %% Actor-Critic
    state = [ones(Tsubj, 1), subj.state];

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
    nd_cost = parameters(7);  % Cost for each action
    cost = exp(nd_cost);

    % Initialize
    D = size(state, 2);  % Dimensionality of state space
    w = w_init * ones(D, 1);  % Critic's state-action value estimates
    theta = theta_init * ones(D, 1);  % Actor's policy parameters
    pred1 = zeros(Tsubj, 1);  % Log probabilities of choices

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
        pred1(t) = 1 / (1 + exp(-product));

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

    %% Logit
    state = [zeros(1,size(subj.state,2)); subj.state(1:end-1, :)];
    lag_choice  = [0;  choice(1:end-1)];       % Lag by 1 step
    lag_outcome = [0; outcome(1:end-1)];
    lag2_choice  = [0;  lag_choice(1:end-1)];  % Lag by 2 steps
    lag2_outcome = [0; lag_outcome(1:end-1)];

    % Interaction terms between lag_outcome and lag_choice
    interaction11 = lag_outcome .* lag_choice;
    interaction12 = lag_outcome .* lag2_choice;
    interaction22 = lag2_outcome .* lag2_choice;
    state1lag2    = state .* lag2_choice;

    % Concatenate the variables to form X
    X = [ones(size(choice, 1), 1), ...
        lag_outcome, lag2_outcome, ...
        lag_choice,  lag2_choice, ...
        interaction11, interaction12, interaction22, ...
        state, state1lag2];
    
    % Extract parameters
    beta = reshape(parameters(9:end), [size(X,2),1]);

    % Compute the linear combination of X and beta
    linear_comb = X * beta;
    % Compute the logistic function for each observation
    pred2 = 1 ./ (1 + exp(-linear_comb));

    %% Hybrid
    weight = parameters(8);
    pred_hybrid = weight * pred1 + (1 - weight) * pred2;

    % Calculate the log likelihood
    loglik = double(choice)' * log(pred_hybrid) + ...
         double(1 - choice)' * log(1-pred_hybrid);

end
