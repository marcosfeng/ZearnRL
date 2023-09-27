function [loglik] = wrapper_function_44(parameters, subj)
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: alerts boosts minutes 
    subj.state = [subj.alerts , subj.boosts , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
