function [loglik] = wrapper_function_34(parameters, subj)
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: minutes 
    subj.state = [subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
