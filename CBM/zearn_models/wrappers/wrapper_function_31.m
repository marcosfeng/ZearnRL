function [loglik] = wrapper_function_31(parameters, subj)
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: activest 
    subj.state = [subj.activest ];
    loglik = actor_critic_model(parameters, subj);
end
