function [loglik] = wrapper_function_73(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: alerts
    subj.outcome = subj.alerts;
    % State variables: activest boosts minutes 
    subj.state = [subj.activest , subj.boosts , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
