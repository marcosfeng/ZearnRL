function [loglik] = wrapper_function_53(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: boosts
    subj.outcome = subj.boosts;
    % State variables: alerts badges 
    subj.state = [subj.alerts , subj.badges ];
    loglik = actor_critic_model(parameters, subj);
end
