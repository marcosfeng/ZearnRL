function [loglik, prob, choice] = baseline_posterior(parameters, subj)
    % Unpack data
    choice = subj.action(1:end);

    % Extract parameter (just one for intercept)
    beta = parameters(1);

    % Compute the probability of choosing action 1
    p1 = 1 ./ (1 + exp(-beta));
    prob = ones(length(choice),1) .* p1;
    prob = [(1 - prob), prob];
    
    % Calculate the log likelihood
    loglik = sum(choice .* log(p1) + (1 - choice) .* log(1 - p1));
end