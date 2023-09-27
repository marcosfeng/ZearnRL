function [loglik] = wrapper_function_68(parameters, subj)
    % Outcome variable: alerts
    subj.outcome = subj.alerts;
    % State variables: badges boosts 
    subj.state = [subj.badges , subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
