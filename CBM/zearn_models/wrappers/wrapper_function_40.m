function [loglik] = wrapper_function_40(parameters, subj)
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: boosts minutes 
    subj.state = [subj.boosts , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
