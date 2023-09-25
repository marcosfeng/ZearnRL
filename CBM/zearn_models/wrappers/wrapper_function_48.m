function [loglik] = wrapper_function_48(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: boosts
    subj.outcome = subj.boosts;
    % State variables: badges 
    subj.state = [subj.badges ];
    loglik = actor_critic_model(parameters, subj);
end
