function [loglik] = wrapper_function_35(parameters, subj)
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: activest alerts 
    subj.state = [subj.activest , subj.alerts ];
    loglik = actor_critic_model(parameters, subj);
end
