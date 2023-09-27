function [loglik] = wrapper_function_4(parameters, subj)
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: minutes 
    subj.state = [subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
