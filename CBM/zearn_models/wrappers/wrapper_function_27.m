function [loglik] = wrapper_function_27(parameters, subj)
    % Outcome variable: minutes
    subj.outcome = subj.minutes;
    % State variables: activest alerts boosts 
    subj.state = [subj.activest , subj.alerts , subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
