function [loglik] = wrapper_function_1(parameters, subj)
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: alerts 
    subj.state = [subj.alerts ];
    loglik = actor_critic_model(parameters, subj);
end
