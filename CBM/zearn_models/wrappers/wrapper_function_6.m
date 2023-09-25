function [loglik] = wrapper_function_6(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: alerts boosts 
    subj.state = [subj.alerts , subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
