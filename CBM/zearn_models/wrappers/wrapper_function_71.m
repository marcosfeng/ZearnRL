function [loglik] = wrapper_function_71(parameters, subj)
    % Outcome variable: alerts
    subj.outcome = subj.alerts;
    % State variables: activest badges boosts 
    subj.state = [subj.activest , subj.badges , subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end