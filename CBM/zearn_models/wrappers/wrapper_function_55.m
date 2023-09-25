function [loglik] = wrapper_function_55(parameters, subj)
    load('../data/all_data.mat');
    % Outcome variable: boosts
    subj.outcome = subj.boosts;
    % State variables: badges minutes 
    subj.state = [subj.badges , subj.minutes ];
    loglik = actor_critic_model(parameters, subj);
end
