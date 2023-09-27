function [loglik] = wrapper_function_45(parameters, subj)
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: activest alerts boosts minutes 
    subj.state = [subj.activest , subj.alerts , subj.boosts , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
