function [loglik] = wrapper_function_21(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: minutes
    subj.outcome = subj.minutes;
    % State variables: activest badges 
    subj.state = [subj.activest , subj.badges ];
    loglik = actor_critic_model(parameters, subj);
end
