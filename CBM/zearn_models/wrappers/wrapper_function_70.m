function [loglik] = wrapper_function_70(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: alerts
    subj.outcome = subj.alerts;
    % State variables: boosts minutes 
    subj.state = [subj.boosts , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
