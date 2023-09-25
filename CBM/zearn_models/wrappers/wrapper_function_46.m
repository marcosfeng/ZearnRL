function [loglik] = wrapper_function_46(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: boosts
    subj.outcome = subj.boosts;
    % State variables: activest 
    subj.state = [subj.activest ];
    loglik = actor_critic_model(parameters, subj);
end
