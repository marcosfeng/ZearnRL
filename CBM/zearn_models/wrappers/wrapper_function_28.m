function [loglik] = wrapper_function_28(parameters, subj)
    % Outcome variable: minutes
    subj.outcome = subj.minutes;
    % State variables: activest badges boosts 
    subj.state = [subj.activest , subj.badges , subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
