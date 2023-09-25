function [loglik] = wrapper_function_38(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: alerts boosts 
    subj.state = [subj.alerts , subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
