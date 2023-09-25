function [loglik] = wrapper_function_33(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: boosts 
    subj.state = [subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
