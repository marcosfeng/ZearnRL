function [loglik] = wrapper_function_25(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: minutes
    subj.outcome = subj.minutes;
    % State variables: badges boosts 
    subj.state = [subj.badges , subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
