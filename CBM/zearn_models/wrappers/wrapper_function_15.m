function [loglik] = wrapper_function_15(parameters, subj)
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: alerts badges boosts minutes 
    subj.state = [subj.alerts , subj.badges , subj.boosts , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
