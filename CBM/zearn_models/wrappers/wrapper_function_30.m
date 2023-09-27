function [loglik] = wrapper_function_30(parameters, subj)
    % Outcome variable: minutes
    subj.outcome = subj.minutes;
    % State variables: activest alerts badges boosts 
    subj.state = [subj.activest , subj.alerts , subj.badges , subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
