function [loglik] = wrapper_function_40(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: boosts minutes 
    subj.state = [subj.boosts , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
