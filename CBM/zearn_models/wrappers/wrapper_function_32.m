function [loglik] = wrapper_function_32(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: alerts 
    subj.state = [subj.alerts ];
    loglik = actor_critic_model(parameters, subj);
end
