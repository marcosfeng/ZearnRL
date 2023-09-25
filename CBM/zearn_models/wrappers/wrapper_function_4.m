function [loglik] = wrapper_function_4(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: minutes 
    subj.state = [subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
