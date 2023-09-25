function [loglik] = wrapper_function_54(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: boosts
    subj.outcome = subj.boosts;
    % State variables: alerts minutes 
    subj.state = [subj.alerts , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
