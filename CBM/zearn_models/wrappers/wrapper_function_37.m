function [loglik] = wrapper_function_37(parameters, subj)
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: activest minutes 
    subj.state = [subj.activest , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
