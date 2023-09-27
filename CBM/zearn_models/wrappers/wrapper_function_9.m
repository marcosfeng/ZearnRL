function [loglik] = wrapper_function_9(parameters, subj)
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: badges minutes 
    subj.state = [subj.badges , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
