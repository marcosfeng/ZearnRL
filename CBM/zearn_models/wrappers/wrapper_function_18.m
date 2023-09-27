function [loglik] = wrapper_function_18(parameters, subj)
    % Outcome variable: minutes
    subj.outcome = subj.minutes;
    % State variables: badges 
    subj.state = [subj.badges ];
    loglik = actor_critic_model(parameters, subj);
end
