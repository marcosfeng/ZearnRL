function [loglik] = wrapper_function_64(parameters, subj)
    % Outcome variable: alerts
    subj.outcome = subj.alerts;
    % State variables: minutes 
    subj.state = [subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
