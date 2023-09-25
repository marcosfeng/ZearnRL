function [loglik] = wrapper_function_35(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: activest alerts 
    subj.state = [subj.activest , subj.alerts ];
    loglik = actor_critic_model(parameters, subj);
end
