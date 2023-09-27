function [loglik] = wrapper_function_58(parameters, subj)
    % Outcome variable: boosts
    subj.outcome = subj.boosts;
    % State variables: activest badges minutes 
    subj.state = [subj.activest , subj.badges , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
