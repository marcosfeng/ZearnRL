function [loglik] = wrapper_function_69(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: alerts
    subj.outcome = subj.alerts;
    % State variables: badges minutes 
    subj.state = [subj.badges , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
