function [loglik, prob, choice] = hybrid_ac_logit_posterior(parameters, subj)
    % Unpack data
    Tsubj = length(subj.action);
    choice = subj.action;  % dummy
    outcome = subj.outcome;
    week = subj.simmed.week;

    %% Actor-Critic
    state = [ones(Tsubj, 1), subj.state];
    D = size(state, 2);  % Dimensionality of state space

    nd_alpha_w = parameters(1);  % Learning rate for w (critic)
    alpha_w = 1/(1+exp(-nd_alpha_w));
    nd_alpha_theta = parameters(2);  % Learning rate for theta (actor)
    alpha_theta = 1/(1+exp(-nd_alpha_theta));
    nd_gamma = parameters(3);  % Discount factor
    gamma = 1/(1+exp(-nd_gamma));
    nd_tau = parameters(4);  % Inverse temperature for softmax action selection
    tau = exp(nd_tau);
    nd_cost = parameters(5);  % Cost for action = 1
    cost = exp(nd_cost);
    theta_init = parameters(6:(5+D));
    w_init = parameters((6+D):(5+2*D));

    % Initialize
    w = w_init';  % Critic's state-action value
    theta = theta_init';  % Actor's policy parameters
    pred1 = zeros(Tsubj, 1);  % Log probabilities of choices

    % Actor: Compute policy (log probability of taking each action)
    % Calculate the product s(t=1) * theta * tau
    product = state(1,:) * theta * tau;
    pred1(1) = 1 / (1 + exp(-product));

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
        pred1(t) = 1 / (1 + exp(-product));
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
    beta = reshape(parameters(((5+2*D)+2):end), [size(X,2),1]);

    % Compute the linear combination of X and beta
    linear_comb = X * beta;
    % Compute the logistic function for each observation
    pred2 = 1 ./ (1 + exp(-linear_comb));

    %% Hybrid
    nd_weight = parameters((5+2*D)+1);
    weight = 1 / (1 + exp(-nd_weight));
    pred_hybrid = weight * pred1 + (1 - weight) * pred2;

    prob = [1-pred_hybrid, pred_hybrid];
    
    % Calculate the log likelihood
    loglik = double(choice)' * log(pred_hybrid) + ...
         double(1 - choice)' * log(1-pred_hybrid);

end
