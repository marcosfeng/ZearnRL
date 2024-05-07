function [loglik] = q_model_2choices(parameters, subj)
    % Unpack data
    C = 2;  % Number of choices
    Tsubj = length(subj.action);
    choice = zeros(Tsubj,C+2);
    choice(:,2:3) = subj.action;
    choice(choice(:,2)==1 & choice(:,3)==1,4) = 1;
    choice(choice(:,4)==1,2:3) = 0;
    choice(choice(:,2)==0 & choice(:,3)==0,1) = 1;
    outcome = subj.outcome;
    week = subj.simmed.week;

    % Extract parameters
    nd_alpha = parameters(1);
    alpha = 1 / (1 + exp(-nd_alpha));
    nd_gamma = parameters(2);  % Discount factor
    gamma = 1/(1+exp(-nd_gamma));
    nd_tau = parameters(3);  % Inverse temperature for softmax action selection
    tau = exp(nd_tau);
    nd_cost = parameters(5:(5+C));  % Cost for actions
    cost = exp(nd_cost);
    ev_init = parameters((6+C):(6+2*C+1)); % Account for simultaneous actions
    % Initialize Q-value for each action
    ev = ev_init*ones(C+1);  % Expected value (Q-value)

    % Save log probability of choice
    log_p = zeros(Tsubj, 1);
    % Efficient way to calculate log likelihood (avoids Inf):
    product = tau * ev;
    log_p(1) = choice(1)*([0, product] - log1p(sum(exp(product))));
   
    % Loop through trials
    for t = 2:Tsubj
        % Read info for the current trial
        a = choice(t);  % Action on this trial
        w_t = week(t);  % Week on this trial
        w_t_prev = week(t-1); % Week on next trial

        if choice(t-1, 1) == 1 % No choice made
            % Update expected value (ev) relative to outside option
            delta = gamma^(double(w_t) - double(w_t_prev)) * outcome(t);
            ev(1) = ev(1) - (alpha * ratio * delta);
            ev(2) = ev(2) - (alpha * (1 - ratio) * delta);
        elseif choice(t-1, 2) == 1 % Choice 1 made
            % Update expected value (ev) for choice 1
            delta = gamma^(double(w_t) - double(w_t_prev)) * ...
                outcome(t) - cost(1) - ev(1);
            ev(1) = ev(1) + (alpha * delta);
        elseif choice(t-1, 3) == 1 % Choice 2 made
            % Update expected value (ev) for choice 2
            delta = gamma^(double(w_t) - double(w_t_prev)) * ...
                outcome(t) - cost(2) - ev(2);
            ev(2) = ev(2) + (alpha * delta);
        elseif choice(t-1, 4) == 1 % Both choices made
            % Update expected value (ev) for both choices
            delta = gamma^(double(w_t) - double(w_t_prev)) * ...
                outcome(t) - cost(1) - cost(2) - ev(1) - ev(2) - ev(3);
            ev(1) = ev(1) + (alpha * ratio * delta);
            ev(2) = ev(2) + (alpha * (1 - ratio) * delta);
            ev(3) = ev(3) + (alpha/ratio * delta);
        end
        
        if choice(t-1,1) == 1
            % Update expected value (ev) relative to outside option
            delta = gamma^(double(w_t) - double(w_t_prev)) * outcome(t);
            ev = ev - (alpha * delta);
            
            % Update expected value (ev) if choice was made
            delta = gamma^(double(w_t) - double(w_t_prev)) * ...
                outcome(t) - cost - ev;
            ev = ev + (alpha * delta);
        elseif choice(t-1,2) == 1
            
        elseif choice(t-1,3) == 1

        elseif choice(t-1,4) == 1

        end


        % Efficient way to calculate log likelihood (avoids Inf):
        product = tau * ev;
        while max(product)-min(product) > 8
            base = min(product);
            product = product - base;
            product = product(~product==0);
        end
        base = min(product);
        product = product - base;
        log_p(1) = choice(1)*(product - log(sum(exp(product))));
    end
    
    % Log-likelihood is defined as the sum of log-probability of choice data
    loglik = sum(log_p);
end
