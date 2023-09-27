function [loglik] = wrapper_function_7(parameters, subj)
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: alerts minutes 
    subj.state = [subj.alerts , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
