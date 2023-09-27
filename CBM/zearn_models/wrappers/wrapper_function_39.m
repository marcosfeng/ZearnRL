function [loglik] = wrapper_function_39(parameters, subj)
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: alerts minutes 
    subj.state = [subj.alerts , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
