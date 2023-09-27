function [loglik] = wrapper_function_20(parameters, subj)
    % Outcome variable: minutes
    subj.outcome = subj.minutes;
    % State variables: activest alerts 
    subj.state = [subj.activest , subj.alerts ];
    loglik = actor_critic_model(parameters, subj);
end
