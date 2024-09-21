rng(37909890)
% © 1998-2023 RANDOM.ORG
% Timestamp: 2023-10-04 18:38:28 UTC

% Load data
fdata = load('../data/full_data.mat');
data = fdata.data;

addpath(fullfile('..','codes'));
addpath(fullfile('..','zearn_codes'));

% Define models and their respective wrapper directories
models = {'q_model', 'logit_model', 'cockburn_model', ...
    'q_model_simple', 'baseline_model'};
wrapper_dirs = {'../zearn_codes/ql_wrappers', ...
                '../zearn_codes/lg_wrappers', ...
                '../zearn_codes/cb_wrappers', ...
                '../zearn_codes/qs_wrappers', ...
                '../zearn_codes/bl_wrappers'};
model_names = {'Q-learning', 'Lau + Glimcher', 'Cockburn et al.', ...
    'Q-learning Simple', 'Baseline'};

num_wrappers = 16;
% Initialize arrays to store results
log_evidence = zeros(length(models) + 1, num_wrappers, length(data));
valid_subj = zeros(length(models) + 1, num_wrappers, length(data));

% Define the prior variance
v = 6.25;

% Determine the number of parameters for the current model
num_parameters = [5,5,6,3,1];

% PCONFIG structure with refined setup (with multiplier)
mult = 4;
pconfig = struct();
pconfig.numinit = min(7 * max(num_parameters), 100) * mult;
pconfig.numinit_med = 70 * mult;
pconfig.numinit_up = 100 * mult;
pconfig.tolgrad = .001001 / mult;
pconfig.tolgrad_liberal = .1 / mult;
pconfig.prior_for_bads = 0;

% Set up parallel pool if not already created
if isempty(gcp('nocreate'))
    parpool;
end
% Loop through each model
for model_num = 1:length(models)
    % Add the correct path for this model
    addpath(wrapper_dirs{model_num});
    
    % Create the prior structure for the model
    prior = struct('mean', zeros(num_parameters(model_num), 1), 'variance', v);
    
    % Loop through each wrapper function
    for wrapper_num = 1:num_wrappers
        % Get the function handle for the current wrapper function
        wrapper_func = str2func(sprintf('wrapper_function_%d', wrapper_num));
        
        % Parallelize over subjects
        parfor subj = 1:length(data)
            % Specify the file-address for saving the output
            fname = sprintf('model_results/lap_%s_%d_subj_%d.mat', ...
                models{model_num}, wrapper_num, subj);
            
            % Run the cbm_lap function for the current wrapper and subject
            cbm_lap(data(subj), wrapper_func, prior, fname, pconfig);
            
            % Load the saved output for this wrapper and subject
            loaded_data = load(fname);
            
            % Check if the subject is valid
            valid_subj(model_num, wrapper_num, subj) = ...
                ~isnan(loaded_data.cbm.math.logdetA) && ...
                ~isinf(loaded_data.cbm.math.logdetA) && ...
                (loaded_data.cbm.math.logdetA ~= 0);
            
            % Store the log evidence for this subject
            if valid_subj(model_num, wrapper_num, subj)
                log_evidence(model_num, wrapper_num, subj) = loaded_data.cbm.output.log_evidence;
            else
                log_evidence(model_num, wrapper_num, subj) = -Inf;
            end
        end
    end
    
    % Remove the path for this model
    rmpath(wrapper_dirs{model_num});
end

% Save success matrix
success = valid_subj;

%% Aggregate results
for model_num = 1:length(models)
    for wrapper_num = 1:num_wrappers
        fname_subjs = arrayfun(@(x) sprintf('model_results/lap_%s_%d_subj_%d.mat', models{model_num}, wrapper_num, x), 1:length(data), 'UniformOutput', false);
        cbm_lap_aggregate(fname_subjs(logical(squeeze(success(model_num, wrapper_num, :)))), ...
                          sprintf('aggr_results/lap_aggr_%s_%d.mat', models{model_num}, wrapper_num));
    end
end

% Calculate total log evidence for each model and wrapper
total_log_evidence = sum(log_evidence, 3);

% Find top 3 wrappers for each model
top_wrappers = zeros(length(models), 3);
top_evidence = zeros(length(models), 3);
for model_num = 1:length(models)
    [sorted_evidence, sorted_indices] = sort(total_log_evidence(model_num, :), 'descend');
    top_wrappers(model_num, :) = sorted_indices(1:3);
    top_evidence(model_num, :) = sorted_evidence(1:3);
end

% Find unique top wrappers across all models
unique_top_wrappers = unique(top_wrappers(:));

%% HBI

% Configure HBI
pconfig = struct();
pconfig.maxiter = 200;

% Prepare arrays to store results
num_combinations = length(unique_top_wrappers) * (length(models) - 1) + 1;
temp_hbi_output = cell(num_combinations, 1);

% Run HBI for each unique top wrapper and model combination
parfor idx = 1:num_combinations
    [wrapper_idx, model_num] = ind2sub([length(unique_top_wrappers), length(models)], idx);
    wrapper_num = unique_top_wrappers(wrapper_idx);
    
    % Prepare model wrapper and file for this combination
    model_wrapper = str2func(sprintf('wrapper_function_%d', wrapper_num));
    model_file = sprintf('aggr_results/lap_aggr_%s_%d.mat', models{model_num}, wrapper_num);
    
    % Add the correct path for this model
    addpath(wrapper_dirs{model_num});

    % Run HBI
    fname_hbi = sprintf('hbi_results/hbi_compare_wrapper_%d_model_%d.mat', wrapper_num, model_num);
    cbm_hbi(data(logical(squeeze(success(model_num, wrapper_num, :)))), ...
            {model_wrapper}, ...
            {model_file}, ...
            fname_hbi, pconfig);

    % Run null model
    cbm_hbi_null(data(logical(squeeze(success(model_num, wrapper_num, :)))), fname_hbi);
    % Load HBI results
    hbi_results = load(fname_hbi);
    temp_hbi_output{idx} = hbi_results.cbm;
      
    % Remove the path for this model
    rmpath(wrapper_dirs{model_num});
end

% Reshape the results to match the original structure
for i = 1:(length(unique_top_wrappers)-1)
    temp_hbi_output{num_combinations + i} = ...
    temp_hbi_output{num_combinations};
end
hbi_output = reshape(temp_hbi_output, length(unique_top_wrappers), length(models));
% Reshape hbi_output to match the original structure
hbi_output = reshape(hbi_output, [], 1);

% Save the results
save('model_comparison_results.mat', 'log_evidence', 'valid_subj', 'top_wrappers', 'top_evidence', 'model_names', 'hbi_output', 'unique_top_wrappers');

%% Posteriors

addpath(fullfile('..','zearn_codes','posterior_wrappers'));

prob = cell(length(data), length(models) * length(unique_top_wrappers));
loglik = nan(length(data), length(models) * length(unique_top_wrappers));
bic = nan(length(data), length(models) * length(unique_top_wrappers));
q_values_all = cell(length(data), length(unique_top_wrappers));

for wrapper_idx = 1:length(unique_top_wrappers)
    wrapper_num = unique_top_wrappers(wrapper_idx);
    
    for model_num = 1:length(models)
        if model_num == 5 && wrapper_num > 4, continue; end
        hbi_fname = sprintf('hbi_results/hbi_compare_wrapper_%d_model_%d.mat', wrapper_num, model_num);
        hbi_model = load(hbi_fname);
                
        % Create the posterior wrapper function name
        posterior_wrapper = str2func(sprintf('posterior_%s_%d', models{model_num}, wrapper_num));
        
        for j = 1:length(data)
            
            if model_num == 1 || model_num == 4
                [loglik(j, (wrapper_idx-1)*length(models) + model_num), ...
                prob{j, (wrapper_idx-1)*length(models) + model_num}, ...
                choice, ...
                q_values] = posterior_wrapper(hbi_model.cbm.output.parameters{1}(j, :), data{j});
                q_values_all{j, wrapper_idx} = q_values;
            else
                [loglik(j, (wrapper_idx-1)*length(models) + model_num), ...
                prob{j, (wrapper_idx-1)*length(models) + model_num}, ...
                choice] = posterior_wrapper(hbi_model.cbm.output.parameters{1}(j, :), data{j});
            end
            bic(j, (wrapper_idx-1)*length(models) + model_num) = ...
                -2 * loglik(j, (wrapper_idx-1)*length(models) + model_num) + ...
                length(hbi_model.cbm.output.parameters{1}(j, :)) * log(length(choice));
        end
    end
end

% Save the results
for wrapper_idx = 1:length(unique_top_wrappers)
    wrapper_num = unique_top_wrappers(wrapper_idx);
    
    for model_num = 1:length(models)
        if model_num == 5 && wrapper_num > 4, continue; end
        hbi_fname = sprintf('hbi_results/hbi_compare_wrapper_%d_model_%d.mat', wrapper_num, model_num);
        load(hbi_fname);
        
        cbm.output.loglik = loglik(:, (wrapper_idx-1)*length(models) + model_num);
        cbm.output.bic = bic(:, (wrapper_idx-1)*length(models) + model_num);
        cbm.output.prob = prob(:, (wrapper_idx-1)*length(models) + model_num);
        cbm.output.choice = data;
        if model_num == 1
            cbm.output.q_values = ...
                q_values_all(:, wrapper_idx);
        end
       
        save(hbi_fname, 'cbm');
    end
end

%% Calculate Posterior Log-Likelihood for Aggregate Models

addpath(fullfile('..','zearn_codes','posterior_wrappers'));

agg_loglik = zeros(length(models), num_wrappers);
agg_bic = zeros(length(models), num_wrappers);

for model_num = 1:length(models)
    for wrapper_num = 1:num_wrappers
        
        % Load aggregate results
        agg_fname = sprintf('aggr_results/lap_aggr_%s_%d.mat', models{model_num}, wrapper_num);
        if ~exist(agg_fname, 'file')
            continue;
        end
        agg_results = load(agg_fname);
        
        % Get aggregate parameters
        agg_params = agg_results.cbm.output.parameters;
        
        % Create the posterior wrapper function name
        posterior_wrapper = str2func(sprintf('posterior_%s_%d', models{model_num}, wrapper_num));
        
        % Calculate log-likelihood for each subject
        subject_logliks = zeros(1, length(data));
        for subj = 1:length(data)      
            [subject_logliks(subj), ~, choice] = posterior_wrapper(agg_params(subj,:), data{subj});
        end

        % Calculate total log-likelihood and BIC
        total_loglik = sum(subject_logliks);
        n_params = size(agg_params, 2);
        n_observations = sum(cellfun(@(x) length(x.simmed.week), ...
            data));
        bic = -2 * subject_logliks' + ...
            n_params * log(cellfun(@(x) length(x.simmed.week), data));
        aic = -2 * subject_logliks' + n_params * 2;
        
        agg_loglik(model_num, wrapper_num) = total_loglik;
        agg_bic(model_num, wrapper_num) = mean(bic);

        % Extract and sum log-likelihoods
        agg_results.cbm.output.loglik = subject_logliks;
        cbm = agg_results.cbm;
        save(agg_fname, 'cbm');
    end
end

% Save results
save('aggregate_model_comparison.mat', 'agg_loglik', 'agg_bic', 'models');
