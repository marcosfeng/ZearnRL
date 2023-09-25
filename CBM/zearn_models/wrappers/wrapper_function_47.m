function [loglik] = wrapper_function_47(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: boosts
    subj.outcome = subj.boosts;
    % State variables: alerts 
    subj.state = [subj.alerts ];
    loglik = actor_critic_model(parameters, subj);
end
