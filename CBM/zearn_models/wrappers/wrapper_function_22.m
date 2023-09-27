function [loglik] = wrapper_function_22(parameters, subj)
    % Outcome variable: minutes
    subj.outcome = subj.minutes;
    % State variables: activest boosts 
    subj.state = [subj.activest , subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
