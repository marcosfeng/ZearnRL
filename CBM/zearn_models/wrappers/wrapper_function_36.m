function [loglik] = wrapper_function_36(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: activest boosts 
    subj.state = [subj.activest , subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
