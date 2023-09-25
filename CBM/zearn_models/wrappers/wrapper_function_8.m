function [loglik] = wrapper_function_8(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: badges boosts 
    subj.state = [subj.badges , subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
