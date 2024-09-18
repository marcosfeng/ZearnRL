% Define variables
state_vars = {'NNDSVD_student1', 'NNDSVD_student2', 'NNDSVD_student3', 'NNDSVD_student4'};
action_vars = {'NNDSVD_teacher1', 'NNDSVD_teacher2', 'NNDSVD_teacher3', 'NNDSVD_teacher4'};

% Create directories for wrapper functions
mkdir('ql_wrappers');
mkdir('lg_wrappers');
mkdir('cb_wrappers');
mkdir('qs_wrappers');
mkdir('bl_wrappers');

%% Q-learning
for i = 1:length(state_vars)
    for j = 1:length(action_vars)
        fname = sprintf('ql_wrappers/wrapper_function_%d.m', ((i-1) * length(action_vars) + j));
        fid = fopen(fname, 'w');
        fprintf(fid, 'function [loglik] = wrapper_function_%d(parameters, subj)\n', ((i-1) * length(action_vars) + j));
        fprintf(fid, '    %% Outcome variable: %s\n', state_vars{i});
        fprintf(fid, '    subj.outcome = subj.%s;\n', state_vars{i});
        fprintf(fid, '    %% Action variable: %s\n', action_vars{j});
        fprintf(fid, '    subj.action = subj.%s;\n', action_vars{j});
        fprintf(fid, '    loglik = q_model(parameters, subj);\n');
        fprintf(fid, 'end\n');
        fclose(fid);
    end
end

%% Q-learning Simple
for i = 1:length(state_vars)
    for j = 1:length(action_vars)
        fname = sprintf('qs_wrappers/wrapper_function_%d.m', ((i-1) * length(action_vars) + j));
        fid = fopen(fname, 'w');
        fprintf(fid, 'function [loglik] = wrapper_function_%d(parameters, subj)\n', ((i-1) * length(action_vars) + j));
        fprintf(fid, '    %% Outcome variable: %s\n', state_vars{i});
        fprintf(fid, '    subj.outcome = subj.%s;\n', state_vars{i});
        fprintf(fid, '    %% Action variable: %s\n', action_vars{j});
        fprintf(fid, '    subj.action = subj.%s;\n', action_vars{j});
        fprintf(fid, '    loglik = q_model_simple(parameters, subj);\n');
        fprintf(fid, 'end\n');
        fclose(fid);
    end
end

%% Baseline
for i = 1:length(state_vars)
    for j = 1:length(action_vars)
        fname = sprintf('bl_wrappers/wrapper_function_%d.m', ((i-1) * length(action_vars) + j));
        fid = fopen(fname, 'w');
        fprintf(fid, 'function [loglik] = wrapper_function_%d(parameters, subj)\n', ((i-1) * length(action_vars) + j));
        fprintf(fid, '    %% Action variable: %s\n', action_vars{j});
        fprintf(fid, '    subj.action = subj.%s;\n', action_vars{j});
        fprintf(fid, '    loglik = baseline_model(parameters, subj);\n');
        fprintf(fid, 'end\n');
        fclose(fid);
    end
end

%% Lau + Glimcher
for i = 1:length(state_vars)
    for j = 1:length(action_vars)
        fname = sprintf('lg_wrappers/wrapper_function_%d.m', ((i-1) * length(action_vars) + j));
        fid = fopen(fname, 'w');
        fprintf(fid, 'function [loglik] = wrapper_function_%d(parameters, subj)\n', ((i-1) * length(action_vars) + j));
        fprintf(fid, '    %% Outcome variable: %s\n', state_vars{i});
        fprintf(fid, '    subj.outcome = subj.%s;\n', state_vars{i});
        fprintf(fid, '    %% Action variable: %s\n', action_vars{j});
        fprintf(fid, '    subj.action = subj.%s;\n', action_vars{j});
        fprintf(fid, '    loglik = logit_model(parameters, subj);\n');
        fprintf(fid, 'end\n');
        fclose(fid);
    end
end

%% Cockburn et al.
for i = 1:length(state_vars)
    for j = 1:length(action_vars)
        fname = sprintf('cb_wrappers/wrapper_function_%d.m', ((i-1) * length(action_vars) + j));
        fid = fopen(fname, 'w');
        
        % Write the main wrapper function
        fprintf(fid, 'function [loglik] = wrapper_function_%d(parameters, subj)\n', ((i-1) * length(action_vars) + j));
        fprintf(fid, '    %% Outcome variable: %s\n', state_vars{i});
        fprintf(fid, '    subj.outcome = subj.%s;\n', state_vars{i});
        fprintf(fid, '    %% Action variable: %s\n', action_vars{j});
        fprintf(fid, '    subj.action = subj.%s;\n', action_vars{j});
        fprintf(fid, '    %% Calculate EV and Uncertainty\n');
        fprintf(fid, '    [subj.ev, subj.sd] = calculate_ev_uncertainty(subj.action, subj.outcome);\n');
        fprintf(fid, '    loglik = cockburn_model(parameters, subj);\n');
        fprintf(fid, 'end\n\n');
        
        % Write the calculate_ev_uncertainty function
        fprintf(fid, 'function [ev, uncertainty] = calculate_ev_uncertainty(action, outcome)\n');
        fprintf(fid, '    v_0 = outcome;\n');
        fprintf(fid, '    v_0(action > 0) = NaN;\n');
        fprintf(fid, '    if isnan(v_0(1))\n');
        fprintf(fid, '        v_0(1) = 0;\n');
        fprintf(fid, '    end\n');
        fprintf(fid, '    v_1 = outcome;\n');
        fprintf(fid, '    v_1(action == 0) = NaN;\n');
        fprintf(fid, '    if isnan(v_1(1))\n');
        fprintf(fid, '        v_1(1) = 0;\n');
        fprintf(fid, '    end\n');
        fprintf(fid, '    \n');
        fprintf(fid, '    ev_0 = cummean_ignore_na(v_0);\n');
        fprintf(fid, '    ev_1 = cummean_ignore_na(v_1);\n');
        fprintf(fid, '    ev = ev_1 - ev_0;\n');
        fprintf(fid, '    \n');
        fprintf(fid, '    uncertainty_0 = cumsd(v_0);\n');
        fprintf(fid, '    uncertainty_1 = cumsd(v_1);\n');
        fprintf(fid, '    uncertainty = uncertainty_1 - uncertainty_0;\n');
        fprintf(fid, 'end\n\n');
        
        % Write the cummean_ignore_na function
        fprintf(fid, 'function result = cummean_ignore_na(x)\n');
        fprintf(fid, '    n = cumsum(~isnan(x));\n');
        fprintf(fid, '    result = cumsum(fillmissing(x, ''constant'', 0)) ./ n;\n');
        fprintf(fid, '    result(n == 0) = 0;\n');
        fprintf(fid, 'end\n\n');
        
        % Write the cumsd function
        fprintf(fid, 'function result = cumsd(x)\n');
        fprintf(fid, '    n = cumsum(~isnan(x));\n');
        fprintf(fid, '    cumsum_x = cumsum(fillmissing(x, ''constant'', 0));\n');
        fprintf(fid, '    cumsum_x2 = cumsum(fillmissing(x.^2, ''constant'', 0));\n');
        fprintf(fid, '    \n');
        fprintf(fid, '    variance = (cumsum_x2 - (cumsum_x.^2) ./ n) ./ (n - 1);\n');
        fprintf(fid, '    variance(isinf(variance)) = NaN;\n');
        fprintf(fid, '    result = sqrt(variance);\n');
        fprintf(fid, '    \n');
        fprintf(fid, '    %% Forward fill NaN values\n');
        fprintf(fid, '    last_valid = NaN;\n');
        fprintf(fid, '    for i = 1:length(result)\n');
        fprintf(fid, '        if ~isnan(result(i))\n');
        fprintf(fid, '            last_valid = result(i);\n');
        fprintf(fid, '        elseif ~isnan(last_valid)\n');
        fprintf(fid, '            result(i) = last_valid;\n');
        fprintf(fid, '        end\n');
        fprintf(fid, '    end\n');
        fprintf(fid, 'end\n');
        
        fclose(fid);
    end
end

%% Posterior Wrappers

models = {'q_model', 'q_model_simple', 'baseline_model', ...
    'logit_model', 'cockburn_model'};

% Create directory for posterior wrappers
mkdir('posterior_wrappers');

for i = 1:length(state_vars)
    for j = 1:length(action_vars)
        wrapper_num = ((i-1) * length(action_vars) + j);
        action_idx = j;
        state_idx = i;
    
        for k = 1:length(models)
            model = models{k};
            fname = sprintf('posterior_wrappers/posterior_%s_%d.m', model, wrapper_num);
            fid = fopen(fname, 'w');
            
            % Write the main posterior function
            if strcmp(model, 'q_model') || strcmp(model, 'q_model_simple')
                fprintf(fid, 'function [loglik, prob, choice, q_values] = posterior_%s_%d(parameters, subj)\n', model, wrapper_num);
            else
                fprintf(fid, 'function [loglik, prob, choice] = posterior_%s_%d(parameters, subj)\n', model, wrapper_num);
            end
            fprintf(fid, '    subj.action = subj.%s;\n', action_vars{action_idx});
            fprintf(fid, '    subj.outcome = subj.%s;\n', state_vars{state_idx});
            
            if strcmp(model, 'cockburn_model')
                fprintf(fid, '    [subj.ev, subj.sd] = calculate_ev_uncertainty(subj.action, subj.outcome);\n');
            end
            
            if strcmp(model, 'q_model') || strcmp(model, 'q_model_simple')
                fprintf(fid, '    [loglik, prob, choice, q_values] = %s_posterior(parameters, subj);\n', strrep(model, '_model', ''));
            else
                fprintf(fid, '    [loglik, prob, choice] = %s_posterior(parameters, subj);\n', strrep(model, '_model', ''));
            end
            fprintf(fid, 'end\n\n');
            
            % Write the calculate_ev_uncertainty function
            fprintf(fid, 'function [ev, uncertainty] = calculate_ev_uncertainty(action, outcome)\n');
            fprintf(fid, '    v_0 = outcome;\n');
            fprintf(fid, '    v_0(action > 0) = NaN;\n');
            fprintf(fid, '    if isnan(v_0(1))\n');
            fprintf(fid, '        v_0(1) = 0;\n');
            fprintf(fid, '    end\n');
            fprintf(fid, '    v_1 = outcome;\n');
            fprintf(fid, '    v_1(action == 0) = NaN;\n');
            fprintf(fid, '    if isnan(v_1(1))\n');
            fprintf(fid, '        v_1(1) = 0;\n');
            fprintf(fid, '    end\n');
            fprintf(fid, '    \n');
            fprintf(fid, '    ev_0 = cummean_ignore_na(v_0);\n');
            fprintf(fid, '    ev_1 = cummean_ignore_na(v_1);\n');
            fprintf(fid, '    ev = ev_1 - ev_0;\n');
            fprintf(fid, '    \n');
            fprintf(fid, '    uncertainty_0 = cumsd(v_0);\n');
            fprintf(fid, '    uncertainty_1 = cumsd(v_1);\n');
            fprintf(fid, '    uncertainty = uncertainty_1 - uncertainty_0;\n');
            fprintf(fid, 'end\n\n');
            
            % Write the cummean_ignore_na function
            fprintf(fid, 'function result = cummean_ignore_na(x)\n');
            fprintf(fid, '    n = cumsum(~isnan(x));\n');
            fprintf(fid, '    result = cumsum(fillmissing(x, ''constant'', 0)) ./ n;\n');
            fprintf(fid, '    result(n == 0) = 0;\n');
            fprintf(fid, 'end\n\n');
            
            % Write the cumsd function
            fprintf(fid, 'function result = cumsd(x)\n');
            fprintf(fid, '    n = cumsum(~isnan(x));\n');
            fprintf(fid, '    cumsum_x = cumsum(fillmissing(x, ''constant'', 0));\n');
            fprintf(fid, '    cumsum_x2 = cumsum(fillmissing(x.^2, ''constant'', 0));\n');
            fprintf(fid, '    \n');
            fprintf(fid, '    variance = (cumsum_x2 - (cumsum_x.^2) ./ n) ./ (n - 1);\n');
            fprintf(fid, '    variance(isinf(variance)) = NaN;\n');
            fprintf(fid, '    result = sqrt(variance);\n');
            fprintf(fid, '    \n');
            fprintf(fid, '    %% Forward fill NaN values\n');
            fprintf(fid, '    last_valid = NaN;\n');
            fprintf(fid, '    for i = 1:length(result)\n');
            fprintf(fid, '        if ~isnan(result(i))\n');
            fprintf(fid, '            last_valid = result(i);\n');
            fprintf(fid, '        elseif ~isnan(last_valid)\n');
            fprintf(fid, '            result(i) = last_valid;\n');
            fprintf(fid, '        end\n');
            fprintf(fid, '    end\n');
            fprintf(fid, 'end\n');
            
            fclose(fid);
        end
    end
end
