function [loglik] = wrapper_function_61(parameters, subj)
    % Outcome variable: alerts
    subj.outcome = subj.alerts;
    % State variables: activest 
    subj.state = [subj.activest ];
    loglik = actor_critic_model(parameters, subj);
end
