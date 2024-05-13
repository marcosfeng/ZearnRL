function [loglik,prob,choice,theta,w] = actor_critic_posterior(parameters, subj)
    % Unpack data
    choice = subj.action; % Binary action
    Tsubj = length(choice);
    outcome = subj.outcome;
    state = [ones(Tsubj, 1), subj.state];
    D = size(state, 2);  % Dimensionality of state space
    week = subj.simmed.week;

    % Extract parameters
    nd_alpha_w = parameters(1);  % Learning rate for w (critic)
    alpha_w = 1/(1+exp(-nd_alpha_w));
    nd_alpha_theta = parameters(2);  % Learning rate for theta (actor)
    alpha_theta = 1/(1+exp(-nd_alpha_theta));
    nd_gamma = parameters(3);  % Discount factor
    gamma = 1/(1+exp(-nd_gamma));
    nd_tau = parameters(4);  % Inverse temperature for softmax action selection
    tau = exp(nd_tau);
    nd_cost = parameters(5);  % Cost for action = 1
    cost = exp(nd_cost);
    theta_init = parameters(6:(5+D));
    w_init = parameters((6+D):end);

    % Initialize
    w = [w_init', ones(D, (Tsubj-1))]; % Critic's state-action value
    theta = [theta_init', ones(D, (Tsubj-1))]; % Actor's policy parameters
    log_p = zeros(Tsubj, 1);  % Log probabilities of choices

    % Actor: Compute policy (log probability of taking each action)
    % Calculate the product s(t=1) * theta * tau
    product = state(1,:) * theta(:,1) * tau;
    % Define a threshold for 'large' theta
    if product < -8
        log_p(1) = product*choice(1)';
    elseif product > 8
        log_p(1) = -product*(1-choice(1))';
    else
        log_p(1) = -log1p(exp(-product))*choice(1)' - ...
            log1p(exp(product))*(1-choice(1))';
    end

    % Loop through trials
    for t = 2:Tsubj
        % End the loop if the week is zero
        if week(t) == 0
            break;
        end
        week_t = week(t);  % Week on this trial
        week_t_1 = week(t - 1); % Week on previous trial
        s_t = state(t,:);  % Current state
        s_t_1 = state(t-1,:); % Previous state
        o = outcome(t);  % Outcome on this trial

        % Critic: Compute TD error (delta)
        if t < Tsubj
            PE = gamma^(double(week_t) - double(week_t_1)) * ...
                (s_t * w(:,t-1)) - (s_t_1 * w(:,t-1));
        else
            PE = 0;  % Terminal state
        end
        delta = o - choice(t-1)*cost + PE;

        % Update weights
        % Derivative of the Log of logistic function = a/(1 + e^(a x))
        theta(:,t) = theta(:,t-1) + alpha_theta * ...
            gamma^(double(week_t_1) - double(week(1))) * ...
            (tau * s_t_1') * (1 + exp(product))^(-1) * delta;
        w(:,t) = w(:,t-1) + delta * s_t_1' * alpha_w';

        % Actor: Compute policy (log probability of taking each action)
        a = choice(t);  % Action on this trial (vector of 0s and 1s)
        % Calculate the product s_t * theta * tau
        product = s_t * theta(:,t) * tau;
        % Define a threshold for 'large' theta
        if product < -8
            log_p(t) = product*a';
        elseif product > 8
            log_p(t) = -product*(1-a)';
        else
            log_p(t) = -log1p(exp(-product))*a' - ...
                log1p(exp(product))*(1-a)';
        end
    end
    % Compute sum of the log-likelihoods
    loglik = double(sum(log_p));
    prob = exp(log_p) .* choice + (1 - exp(log_p)) .* (1 - choice);
    prob = [1-prob, prob];
end
