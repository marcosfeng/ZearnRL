function [loglik] = wrapper_function_42(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: activest alerts minutes 
    subj.state = [subj.activest , subj.alerts , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
