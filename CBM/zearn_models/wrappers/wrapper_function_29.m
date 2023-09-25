function [loglik] = wrapper_function_29(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: minutes
    subj.outcome = subj.minutes;
    % State variables: alerts badges boosts 
    subj.state = [subj.alerts , subj.badges , subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
