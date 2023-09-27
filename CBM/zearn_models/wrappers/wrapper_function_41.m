function [loglik] = wrapper_function_41(parameters, subj)
    % Outcome variable: badges
    subj.outcome = subj.badges;
    % State variables: activest alerts boosts 
    subj.state = [subj.activest , subj.alerts , subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
