function [loglik] = wrapper_function_12(parameters, subj)
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: alerts badges minutes 
    subj.state = [subj.alerts , subj.badges , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
