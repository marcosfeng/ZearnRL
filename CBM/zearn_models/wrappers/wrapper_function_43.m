function [loglik] = wrapper_function_43(parameters, subj)
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: activest boosts minutes 
    subj.state = [subj.activest , subj.boosts , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
