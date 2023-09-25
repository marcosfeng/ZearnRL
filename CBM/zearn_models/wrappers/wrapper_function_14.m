function [loglik] = wrapper_function_14(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: badges boosts minutes 
    subj.state = [subj.badges , subj.boosts , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
