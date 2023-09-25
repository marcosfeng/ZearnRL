function [loglik] = wrapper_function_17(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: minutes
    subj.outcome = subj.minutes;
    % State variables: alerts 
    subj.state = [subj.alerts ];
    loglik = actor_critic_model(parameters, subj);
end
