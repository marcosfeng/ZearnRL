function [loglik] = wrapper_function_8(parameters, subj)
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: badges boosts 
    subj.state = [subj.badges , subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
