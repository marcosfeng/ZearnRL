function [loglik] = wrapper_function_72(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: alerts
    subj.outcome = subj.alerts;
    % State variables: activest badges minutes 
    subj.state = [subj.activest , subj.badges , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
