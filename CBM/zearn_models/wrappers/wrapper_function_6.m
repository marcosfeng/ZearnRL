function [loglik] = wrapper_function_6(parameters, subj)
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: alerts boosts 
    subj.state = [subj.alerts , subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
