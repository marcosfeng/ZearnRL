function [loglik] = wrapper_function_24(parameters, subj)
    % Outcome variable: minutes
    subj.outcome = subj.minutes;
    % State variables: alerts boosts 
    subj.state = [subj.alerts , subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
