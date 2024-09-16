function [loglik] = baseline_model(parameters, subj)
    % Unpack data
    choice = subj.action(1:end);

    % Extract parameter (just one for intercept)
    X = [ones(size(choice, 1), 1)];
    beta = reshape(parameters, [size(X,2),1]);
    linear_comb = X * beta;

    % Compute log-likelihood using log1p(exp()) for consistency
    loglik = choice' * (-log1p(exp(-linear_comb))) + ...
        (1 - choice)' * (-log1p(exp(linear_comb)));
end