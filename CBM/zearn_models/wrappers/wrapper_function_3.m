function [loglik] = wrapper_function_3(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: activest
    subj.outcome = subj.activest;
    % State variables: boosts 
    subj.state = [subj.boosts ];
    loglik = actor_critic_model(parameters, subj);
end
