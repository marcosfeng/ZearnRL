function [loglik] = wrapper_function_65(parameters, subj)
    % Outcome variable: alerts
    subj.outcome = subj.alerts;
    % State variables: activest badges 
    subj.state = [subj.activest , subj.badges ];
    loglik = actor_critic_model(parameters, subj);
end
