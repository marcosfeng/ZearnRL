function [loglik] = wrapper_function_13(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: alerts boosts minutes 
    subj.state = [subj.alerts , subj.boosts , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
