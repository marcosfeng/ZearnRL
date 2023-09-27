function [loglik] = wrapper_function_49(parameters, subj)
    % Outcome variable: boosts
    subj.outcome = subj.boosts;
    % State variables: minutes 
    subj.state = [subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
