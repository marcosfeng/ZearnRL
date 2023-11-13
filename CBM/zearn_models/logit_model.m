function [loglik] = logit_model(parameters, subj)
    
    % Unpack data
    choice = subj.actions(1:end, :);
    outcome = subj.minutes(1:end);
    alerts = subj.alerts(2:end, :);
    boosts = subj.boosts(2:end, :);
    C = size(choice, 2);

    lag_choice  = [choice(1:end-1, :)];  % Lag by 1 step
    lag_outcome = [outcome(1:end-1)];  % Lag by 1 step

    choice = subj.actions(2:end, :);

    % Concatenate the variables to form X
    X = [ones(size(choice, 1), 1), lag_outcome, ...
        alerts, boosts];

    % Extract parameters
    beta = reshape(parameters, (size(X,2)+1), C);

    loglik = 0;
    for c = C
        % Compute the linear combination of X and beta
        linear_comb = [X, lag_choice(:,c)] * beta(:, c);
        % Compute the logistic function for each observation
        logistic_prob = 1 ./ (1 + exp(-linear_comb));
        % Calculate the log likelihood
        p = choice(:, c) .* log(logistic_prob) + ...
            (1 - choice(:, c)) .* log(1 - logistic_prob);
        
        % Update log likelihood
        loglik = loglik + sum(p);
    end

    % Compute log-likelihood
    loglik = double(loglik);
end
