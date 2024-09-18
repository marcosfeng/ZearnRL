function [loglik, prob, choice, q_values] = posterior_q_model_simple_9(parameters, subj)
    subj.action = subj.NNDSVD_teacher1;
    subj.outcome = subj.NNDSVD_student3;
    [loglik, prob, choice, q_values] = q_simple_posterior(parameters, subj);
end

function [ev, uncertainty] = calculate_ev_uncertainty(action, outcome)
    v_0 = outcome;
    v_0(action > 0) = NaN;
    if isnan(v_0(1))
        v_0(1) = 0;
    end
    v_1 = outcome;
    v_1(action == 0) = NaN;
    if isnan(v_1(1))
        v_1(1) = 0;
    end
    
    ev_0 = cummean_ignore_na(v_0);
    ev_1 = cummean_ignore_na(v_1);
    ev = ev_1 - ev_0;
    
    uncertainty_0 = cumsd(v_0);
    uncertainty_1 = cumsd(v_1);
    uncertainty = uncertainty_1 - uncertainty_0;
end

function result = cummean_ignore_na(x)
    n = cumsum(~isnan(x));
    result = cumsum(fillmissing(x, 'constant', 0)) ./ n;
    result(n == 0) = 0;
end

function result = cumsd(x)
    n = cumsum(~isnan(x));
    cumsum_x = cumsum(fillmissing(x, 'constant', 0));
    cumsum_x2 = cumsum(fillmissing(x.^2, 'constant', 0));
    
    variance = (cumsum_x2 - (cumsum_x.^2) ./ n) ./ (n - 1);
    variance(isinf(variance)) = NaN;
    result = sqrt(variance);
    
    % Forward fill NaN values
    last_valid = NaN;
    for i = 1:length(result)
        if ~isnan(result(i))
            last_valid = result(i);
        elseif ~isnan(last_valid)
            result(i) = last_valid;
        end
    end
end
