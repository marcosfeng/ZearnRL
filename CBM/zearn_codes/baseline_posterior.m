function [loglik, prob, choice] = baseline_posterior(parameters, subj)
    % Unpack data
    choice = subj.action(1:end);

    % Extract parameter (just one for intercept)
    X = [ones(size(choice, 1), 1)];
    beta = reshape(parameters, [size(X,2),1]);
    linear_comb = X * beta;

    yes_prob = -log1p(exp(-linear_comb));
    no_prob = -log1p(exp(linear_comb));

    % Compute log-likelihood using log1p(exp()) for consistency
    loglik = choice' * yes_prob + ...
        (1 - choice)' * no_prob;

    % Compute the probability of choosing action 1
    prob = [exp(no_prob), exp(yes_prob)];
end