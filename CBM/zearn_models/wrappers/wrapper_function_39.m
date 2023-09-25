function [loglik] = wrapper_function_39(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: alerts minutes 
    subj.state = [subj.alerts , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
