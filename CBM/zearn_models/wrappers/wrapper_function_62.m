function [loglik] = wrapper_function_62(parameters, subj)
    % Outcome variable: alerts
    subj.outcome = subj.alerts;
    % State variables: badges 
    subj.state = [subj.badges ];
    loglik = actor_critic_model(parameters, subj);
end