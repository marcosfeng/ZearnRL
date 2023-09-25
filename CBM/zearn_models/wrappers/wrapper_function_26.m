function [loglik] = wrapper_function_26(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: minutes
    subj.outcome = subj.minutes;
    % State variables: activest alerts badges 
    subj.state = [subj.activest , subj.alerts , subj.badges ];
    loglik = actor_critic_model(parameters, subj);
end
