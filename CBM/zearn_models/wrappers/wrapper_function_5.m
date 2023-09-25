function [loglik] = wrapper_function_5(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: alerts badges 
    subj.state = [subj.alerts , subj.badges ];
    loglik = actor_critic_model(parameters, subj);
end
