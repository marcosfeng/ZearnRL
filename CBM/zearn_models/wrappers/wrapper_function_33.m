function [loglik] = wrapper_function_33(parameters, subj)
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: boosts 
    subj.state = [subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
