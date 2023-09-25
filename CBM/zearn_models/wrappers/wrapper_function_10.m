function [loglik] = wrapper_function_10(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: boosts minutes 
    subj.state = [subj.boosts , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
