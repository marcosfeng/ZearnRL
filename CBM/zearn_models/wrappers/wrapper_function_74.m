function [loglik] = wrapper_function_74(parameters, subj)
    % Outcome variable: alerts
    subj.outcome = subj.alerts;
    % State variables: badges boosts minutes 
    subj.state = [subj.badges , subj.boosts , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
