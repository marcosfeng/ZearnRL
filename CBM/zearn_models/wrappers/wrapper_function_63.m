function [loglik] = wrapper_function_63(parameters, subj)
    % Outcome variable: alerts
    subj.outcome = subj.alerts;
    % State variables: boosts 
    subj.state = [subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
