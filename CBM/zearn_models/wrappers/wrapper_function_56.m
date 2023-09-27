function [loglik] = wrapper_function_56(parameters, subj)
    % Outcome variable: boosts
    subj.outcome = subj.boosts;
    % State variables: activest alerts badges 
    subj.state = [subj.activest , subj.alerts , subj.badges ];
    loglik = actor_critic_model(parameters, subj);
end
