function [loglik] = wrapper_function_67(parameters, subj)
    % Outcome variable: alerts
    subj.outcome = subj.alerts;
    % State variables: activest minutes 
    subj.state = [subj.activest , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end