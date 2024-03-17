function [loglik] = hybrid_ql_logit(parameters, subj)
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

    %% Logit
    lag_choice  = [0;  choice(1:end-1)];       % Lag by 1 step
    lag_outcome = [0; outcome(1:end-1)];
    lag2_choice  = [0;  lag_choice(1:end-1)];  % Lag by 2 steps
    lag2_outcome = [0; lag_outcome(1:end-1)];

    % Interaction terms between lag_outcome and lag_choice
    interaction11 = lag_outcome .* lag_choice;
    interaction12 = lag_outcome .* lag2_choice;
    interaction22 = lag2_outcome .* lag2_choice;

    % Concatenate the variables to form X
    X = [ones(size(choice, 1), 1), ...
        lag_outcome, lag2_outcome, ...
        lag_choice,  lag2_choice, ...
        interaction11, interaction12, interaction22];
    
    % Extract parameters
    beta = reshape(parameters(6:end), [size(X,2),1]);

    % Compute the linear combination of X and beta
    linear_comb = X * beta;
    % Compute the logistic function for each observation
    pred2 = 1 ./ (1 + exp(-linear_comb));

    %% Hybrid
    weight = parameters(5);
    pred_hybrid = weight * pred1 + (1 - weight) * pred2;

    % Calculate the log likelihood
    loglik = double(choice)' * log(pred_hybrid) + ...
         double(1 - choice)' * log(1-pred_hybrid);
    
end
