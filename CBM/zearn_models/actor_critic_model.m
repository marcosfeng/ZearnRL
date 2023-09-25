function [loglik] = actor_critic_model(parameters, subj)
    % Extract parameters
    alpha_w = parameters(1);  % Learning rate for w (critic)
    alpha_theta = parameters(2);  % Learning rate for theta (actor)
    gamma = parameters(3);  % Discount factor
    tau = parameters(4);  % Inverse temperature for softmax action selection
    cost = parameters(5:end);  % Cost for each action
    
    % Unpack data
    Tsubj = length(subj.actions);
    choice = subj.actions;
    outcome = subj.outcome;
    state = [ones(Tsubj, 1), subj.state];
    week = subj.simmed.week;
    
    % Initialize
    C = size(choice, 2);  % Number of choices
    D = size(state, 2);  % Dimensionality of state space
    w = zeros(D, C);  % Critic's state-action value estimates
    theta = zeros(D, C);  % Actor's policy parameters
    p = zeros(size(subj.actions, 1), 1);  % Log probabilities of choices
    
    % Loop through trials
    for t = 1:Tsubj
        % End the loop if the week is zero
        w_t = week(t);  % Week on this trial
        if week(t) == 0
            break;
        end
        s = state(t, :);  % Current state (now a vector)
        a = choice(t, :);  % Action on this trial (vector of 0s and 1s)
        o = outcome(t);  % Outcome on this trial
        
        % Actor: Compute policy (probability of taking each action)
        logits = s * theta;
        % Clipping to avoid numerical overflow
        policy = max(1 ./ (1 + exp(-logits * tau)), 1e-100);
        
        % Compute log probability of the chosen actions
        p(t) = sum(log(policy(a == 1)));
        
        % Critic: Compute TD error (delta)
        if t < Tsubj
            w_t_next = week(t + 1); % Week on next trial
            s_next = state(t + 1, :);  % State on next trial
            PE = gamma^(double(w_t_next) - double(w_t)) * (s_next * w) - (s * w);
        else
            PE = 0;  % Terminal state
        end
        delta = (o - cost) - PE;
        
        % Update weights
        theta = theta + alpha_theta * s' * delta;
        w = w + alpha_w * s' * delta;
    end
    
    % Compute log-likelihood
    loglik = sum(p);
end
