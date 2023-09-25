function [loglik] = wrapper_function_1(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: alerts 
    subj.state = [subj.alerts ];
    loglik = actor_critic_model(parameters, subj);
end
