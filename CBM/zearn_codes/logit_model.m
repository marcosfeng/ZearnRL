function [loglik] = logit_model(parameters, subj)
    % Unpack data
    choice = subj.action(1:end);
    outcome = subj.outcome(1:end);

    lag_choice  = [0;  choice(1:end-1)];       % Lag by 1 step
    lag_outcome = [0; outcome(1:end-1)];
    lag2_choice  = [0;  lag_choice(1:end-1)];  % Lag by 2 steps
    lag2_outcome = [0; lag_outcome(1:end-1)];

    % % Interaction terms between lag_outcome and lag_choice
    % interaction11 = lag_outcome .* lag_choice;
    % interaction12 = lag_outcome .* lag2_choice;
    % interaction22 = lag2_outcome .* lag2_choice;

    % Concatenate the variables to form X
    X = [ones(size(choice, 1), 1), ...
        lag_outcome, lag2_outcome, ...
        lag_choice,  lag2_choice];
        % interaction11, interaction12, interaction22];

    % Extract parameters
    beta = reshape(parameters, [size(X,2),1]);

    % Compute the linear combination of X and beta
    linear_comb = X * beta;
    % Compute the logistic function for each observation
    yes_prob = -log1p(exp(-linear_comb));
    no_prob = -log1p(exp(linear_comb));
    % Calculate the log likelihood
    p = choice'* yes_prob + ...
        (1 - choice)' * no_prob;

    % Compute log-likelihood
    loglik = double(p);
end
