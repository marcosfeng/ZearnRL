function [loglik] = wrapper_function_19(parameters, subj)
    % Outcome variable: minutes
    subj.outcome = subj.minutes;
    % State variables: boosts 
    subj.state = [subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
