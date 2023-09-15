function [loglik] = q_learning_model(parameters, subj)
    % Extract parameters
    nd_alpha = parameters(1);
    alpha = 1 / (1 + exp(-nd_alpha));
    
    nd_gamma = parameters(2);
    gamma = exp(nd_gamma);
    
    nd_tau = parameters(3);
    tau = exp(nd_tau);
    
    cost = parameters(4:end);  % Assuming cost is a vector
    
    % Unpack data
    Tsubj = subj.Tsubj;
    choice = subj.choice;  % 1 for action=1 and 2 for action=2, etc.
    outcome = subj.outcome;  % 1 for outcome=1 and 0 for outcome=0
    state = subj.state;
    
    % Initialize Q-value for each action
    C = length(cost);  % Number of choices
    S = max(state(:));  % Number of states
    ev = zeros(C, S);  % Expected value (Q-value) for both actions initialized at 0
    
    % To save probability of choice. Currently NaNs, will be filled below
    p = nan(Tsubj, 1);
    
    % Loop through trials
    for t = 1:Tsubj
        % Probability of each action
        prob_choice = exp(tau .* ev(:, state(t))) ./ sum(exp(tau .* ev(:, state(t))));
        
        % Read info for the current trial
        a = choice(t, :);  % Action on this trial
        o = outcome(t);  % Outcome on this trial
        s = state(t);  % State on this trial
        
        % Store probability of the chosen action
        for j = 1:C
            if a(j) == 1
                p(t) = prob_choice(j);
                
                % Update expected value (ev) if choice was made
                delta = gamma * (o - cost(j)) - ev(j, s);  % Prediction error
                ev(j, s) = ev(j, s) + (alpha * delta);
            end
        end
    end
    
    % Log-likelihood is defined as the sum of log-probability of choice data
    % (given the parameters).
    loglik = sum(log(p + eps));
end
