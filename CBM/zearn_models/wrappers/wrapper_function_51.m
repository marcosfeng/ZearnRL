function [loglik] = wrapper_function_51(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: boosts
    subj.outcome = subj.boosts;
    % State variables: activest badges 
    subj.state = [subj.activest , subj.badges ];
    loglik = actor_critic_model(parameters, subj);
end
