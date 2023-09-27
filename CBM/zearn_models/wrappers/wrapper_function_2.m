function [loglik] = wrapper_function_2(parameters, subj)
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: badges 
    subj.state = [subj.badges ];
    loglik = actor_critic_model(parameters, subj);
end
