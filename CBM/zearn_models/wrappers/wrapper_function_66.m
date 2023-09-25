function [loglik] = wrapper_function_66(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: alerts
    subj.outcome = subj.alerts;
    % State variables: activest boosts 
    subj.state = [subj.activest , subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
