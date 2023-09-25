function [loglik] = wrapper_function_57(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: boosts
    subj.outcome = subj.boosts;
    % State variables: activest alerts minutes 
    subj.state = [subj.activest , subj.alerts , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
