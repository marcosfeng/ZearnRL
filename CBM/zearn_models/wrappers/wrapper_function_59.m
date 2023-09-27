function [loglik] = wrapper_function_59(parameters, subj)
    % Outcome variable: boosts
    subj.outcome = subj.boosts;
    % State variables: alerts badges minutes 
    subj.state = [subj.alerts , subj.badges , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
