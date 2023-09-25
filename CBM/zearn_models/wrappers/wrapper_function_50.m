function [loglik] = wrapper_function_50(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: boosts
    subj.outcome = subj.boosts;
    % State variables: activest alerts 
    subj.state = [subj.activest , subj.alerts ];
    loglik = actor_critic_model(parameters, subj);
end
