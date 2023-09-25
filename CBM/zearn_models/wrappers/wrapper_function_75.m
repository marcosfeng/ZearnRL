function [loglik] = wrapper_function_75(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: alerts
    subj.outcome = subj.alerts;
    % State variables: activest badges boosts minutes 
    subj.state = [subj.activest , subj.badges , subj.boosts , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
