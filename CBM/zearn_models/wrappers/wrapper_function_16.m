function [loglik] = wrapper_function_16(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: minutes
    subj.outcome = subj.minutes;
    % State variables: activest 
    subj.state = [subj.activest ];
    loglik = actor_critic_model(parameters, subj);
end
