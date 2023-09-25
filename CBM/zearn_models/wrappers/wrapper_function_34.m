function [loglik] = wrapper_function_34(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: minutes 
    subj.state = [subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
