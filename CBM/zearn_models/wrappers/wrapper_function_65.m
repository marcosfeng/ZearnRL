function [loglik] = wrapper_function_65(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: alerts
    subj.outcome = subj.alerts;
    % State variables: activest badges 
    subj.state = [subj.activest , subj.badges ];
    loglik = actor_critic_model(parameters, subj);
end
