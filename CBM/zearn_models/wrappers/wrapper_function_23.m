function [loglik] = wrapper_function_23(parameters, subj)
    % Outcome variable: minutes
    subj.outcome = subj.minutes;
    % State variables: alerts badges 
    subj.state = [subj.alerts , subj.badges ];
    loglik = actor_critic_model(parameters, subj);
end
