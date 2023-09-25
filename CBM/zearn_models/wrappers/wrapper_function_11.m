function [loglik] = wrapper_function_11(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: alerts badges boosts 
    subj.state = [subj.alerts , subj.badges , subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
