function [loglik] = q_state_model(parameters, subj)
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
    state = (subj.alerts >= subj.medianAlerts) + 1; % Global median (change accordingly)

    % Initialize Q-value for each action
    C = length(cost);  % Number of choices
    S = max(state(:));  % Number of states
    ev = zeros(C, S);  % Expected value (Q-value) for both actions initialized at 0
    K = 4;

    % To save log probability of choice. Currently zeros, will be filled below
    log_p = zeros(Tsubj, 1);

    % Loop through trials
    for t = 1:Tsubj
        % Read info for the current trial
        a = choice(t, :);  % Action on this trial
        s = state(t);  % State on this trial

        % Kernel reward calculation
        for j = 1:C
            % Compute log probability of the chosen action using logistic function
            logit_val = tau * ev(j, state(t));
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
                    jaccard_sim = 0;
                    % Check if the action is the same in the past
                    if choice(t - t_past, j) == a(j)
                        jaccard_sim = 0.5;
                        % Check if the state is also the same in the past
                        if state(t - t_past) == s
                            jaccard_sim = jaccard_sim + 0.5;
                        end
                        % Calculate kernelized reward
                        ker_reward = ker_reward + ...
                                     gamma^(1 + t - (t - t_past)) * ...
                                     jaccard_sim * outcome(t - t_past);
                        ker_norm = ker_norm + jaccard_sim;
                    end
                end
                if ker_norm > 0
                    ker_reward = ker_reward / ker_norm;
                else
                    ker_reward = 0;
                end

                % Update expected value (ev) if choice was made
                delta = (ker_reward - cost(j)) - ev(j, s);  % Prediction error with kernelized reward
                ev(j, s) = ev(j, s) + (alpha * delta);
            end
        end
    end
    
    % Log-likelihood is defined as the sum of log-probability of choice data
    loglik = sum(log_p);
end
