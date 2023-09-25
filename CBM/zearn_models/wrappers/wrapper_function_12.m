function [loglik] = wrapper_function_12(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: alerts badges minutes 
    subj.state = [subj.alerts , subj.badges , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
