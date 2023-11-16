function [loglik] = logit_model(parameters, subj)
    
    % Unpack data
    choice = subj.actions(1:end, :);
    outcome = subj.badges(1:end);
    alerts = [0; subj.alerts(2:end, :)];
    C = size(choice, 2);

    lag_choice  = [zeros(1,C); choice(1:end-1, :)];  % Lag by 1 step
    lag_outcome = [0; outcome(1:end-1)];  % Lag by 1 step

    % Concatenate the variables to form X
    X = [ones(size(choice, 1), 1), lag_outcome, alerts];

    % Extract parameters
    beta = reshape(parameters, (size(X,2)+1), C);

    loglik = 0;
    for c = 1:C
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
