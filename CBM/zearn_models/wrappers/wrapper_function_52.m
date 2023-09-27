function [loglik] = wrapper_function_52(parameters, subj)
    % Outcome variable: boosts
    subj.outcome = subj.boosts;
    % State variables: activest minutes 
    subj.state = [subj.activest , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
