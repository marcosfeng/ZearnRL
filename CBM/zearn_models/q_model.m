function [loglik] = q_model(parameters, subj)
    % Extract parameters
    nd_alpha = parameters(1);
    alpha = 1 / (1 + exp(-nd_alpha));
    nd_gamma = parameters(2);
    gamma = 1 / (1 + exp(-nd_gamma));
    nd_tau = parameters(3);
    tau = exp(nd_tau);
    nd_cost = parameters(4:end);  % cost is a vector
    cost = exp(nd_cost);
    
    % Unpack data
    Tsubj = length(subj.actions);
    choice = subj.actions;  % 1 for action=1 and 2 for action=2, etc.
    outcome = subj.badges;  % 1 for outcome=1 and 0 for outcome=0

    % Initialize Q-value for each action
    C = length(cost);  % Number of choices
    ev = zeros(C);  % Expected value (Q-value) for both actions initialized at 0
    K = 4;

    % To save log probability of choice. Currently zeros, will be filled below
    log_p = zeros(Tsubj, 1);

    % Loop through trials
    for t = 1:Tsubj
        % Read info for the current trial
        a = choice(t, :);  % Action on this trial

        % Kernel reward calculation
        for j = 1:C
            % Compute log probability of the chosen action using logistic function
            logit_val = tau * ev(j);
            if logit_val < -8
                log_p(t) = log_p(t) + a(j)*(-logit_val);
            elseif logit_val > 8
                log_p(t) = log_p(t) + (1 - a(j))*(logit_val);
            else
                log_p(t) = log_p(t) + ...
                    a(j)*(-log1p(exp(-logit_val))) + ...
                    (1 - a(j))*(-log1p(exp(logit_val)));
            end

            if a(j) == 1
                ker_reward = 0;
                ker_norm = 0;
                for t_past = 1:min(t-1, K)
                    % Check if the action is the same in the past
                    if choice(t - t_past, j) == a(j)
                        % Calculate kernelized reward
                        ker_reward = ker_reward + ...
                                     gamma^(1 + t - (t - t_past)) * ...
                                     outcome(t - t_past);
                        ker_norm = ker_norm + 1;
                    end
                end
                if ker_norm > 0
                    ker_reward = ker_reward / ker_norm;
                else
                    ker_reward = 0;
                end

                % Update expected value (ev) if choice was made
                delta = (ker_reward - cost(j)) - ev(j);  % Prediction error with kernelized reward
                ev(j) = ev(j) + (alpha * delta);
            end
        end
    end
    
    % Log-likelihood is defined as the sum of log-probability of choice data
    loglik = sum(log_p);
end
