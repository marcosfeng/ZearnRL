rng(37909890)
% © 1998-2023 RANDOM.ORG
% Timestamp: 2023-10-04 18:38:28 UTC

% Add paths
addpath(fullfile('..','codes'));
addpath(fullfile('..','zearn_codes'));
addpath(fullfile('..','zearn_codes','ac_wrappers'));
addpath(fullfile('ac_subj_results'));

% Load the common data for all datasets
load("top10_ac.mat");
fdata = load('../data/all_data.mat');
data  = fdata.data;

%% Re-estimate top models with more precision

models = cell(1,10);
fname = cell(1,10);

% Create the prior structure for your new model
v = 6.25/2;
num_parameters = 7;
% Create the prior structure from previous estimation
prior_ac = struct('mean', zeros(num_parameters, 1), 'variance', v);
for i = 1:10
    models{i} = str2func(sprintf('wrapper_function_%d', top10_indices(i)));
    fname{i} = sprintf('ac_subj_results/lap_ac_%d.mat', top10_indices(i));

    loaded_data = load(fname{i});
    prior_ac.mean = prior_ac.mean + ...
        mean(loaded_data.cbm.output.parameters,1)';
    fname{i} = sprintf('ac_refine/refine_ac_%d.mat', top10_indices(i));
end
prior_ac.mean = prior_ac.mean/10;

% Populate the top models and their corresponding fcbm_maps
% Create the PCONFIG struct
pconfig = struct();
pconfig.numinit = min(140*num_parameters, 2000);
pconfig.numinit_med = 2000;
pconfig.numinit_up = 20000;
pconfig.tolgrad = 5e-5;
pconfig.tolgrad_liberal = 0.005;
% Initialize a parallel pool if it doesn't already exist
if isempty(gcp('nocreate'))
    parpool;
end
parfor i = 1:10
    % Run the cbm_lap function for your new model
    cbm_lap(data, models{i}, prior_ac, fname{i}, pconfig);
end

%% Histograms by valid log evidence

model_desc = cell(size(fname));
valid_subj_all = ones(1,295);
% Loop over each file name to construct the model description
for i = 1:length(fname)
    loaded_data = load(fname{i});

    % 1) Create a logical index for valid subjects
    valid_subjects = ~isnan(loaded_data.cbm.math.logdetA) ...
        & ~isinf(loaded_data.cbm.math.logdetA) ...
        & (loaded_data.cbm.math.logdetA ~= 0);
    % Calculate the mean and SD of logdetA for valid subjects
    mean_logdetA = mean(loaded_data.cbm.math.logdetA(valid_subjects));
    std_logdetA = std(loaded_data.cbm.math.logdetA(valid_subjects));
    % Add the condition for logdetA values within 3 SDs of the mean
    valid_subjects = valid_subjects & ...
        (abs(loaded_data.cbm.math.logdetA - mean_logdetA) <= ...
        3 * std_logdetA);
    valid_subj_all = valid_subj_all & valid_subjects;
end

for i = 1:length(fname)
    % 2) Create a cell array for the model descriptions
    % Extract the number from the filename
    num = regexp(fname{i}, '\d+', 'match');
    num = num{1}; % Assuming there's only one number in the filename
    
    % Construct the path to the corresponding .m file
    wrapper_filename = fullfile('..', 'zearn_codes', ...
        'ac_wrappers', ['wrapper_function_' num '.m']);
    
    % Read the .m file
    file_contents = fileread(wrapper_filename);
    
    % Extract the outcome variable name
    outcome_match = regexp(file_contents, ...
        'subj\.outcome\s*=\s*subj\.(\w+);', 'tokens');
    % Extract the outcome variable name
    action_match  = regexp(file_contents, ...
        'subj\.action\s*=\s*subj\.(\w+);', 'tokens');
    % Extract the entire string of state variables
    state_match   = regexp(file_contents, ...
        'subj\.state\s*=\s*\[(subj\.\w+\s*(?:,\s*subj\.\w+\s*)*)\];', 'match');
    
    if ~isempty(outcome_match) && ~isempty(state_match)
        outcome_var = outcome_match{1}{1};
        action_var  = action_match{1}{1};
        state_var_string = state_match{1};
        
        % Now split the state variable string into individual variables
        state_vars = regexp(state_var_string, 'subj\.(\w+)', 'tokens');
        % Flatten the nested cell array resulting from regexp
        state_vars = [state_vars{2:length(state_vars)}];
        
        % Construct the model description
        model_desc{i} = sprintf('Action: %s, Outcome: %s, States: %s', ...
            (action_var), (outcome_var), strjoin((state_vars), ', '));
    else
        % If the pattern is not found, use a placeholder
        model_desc{i} = 'Outcome and States not found in wrapper';
    end
end

% Convert the log evidence to non-scientific notation using a cell array
log_evidence = zeros(1, 10);
log_evidence_non_sci = cell(size(log_evidence));
for i = 1:length(log_evidence)
    loaded_data = load(fname{i});
    log_evidence(i) = ...
        sum(loaded_data.cbm.output.log_evidence(valid_subj_all));
    log_evidence_non_sci{i} = num2str(log_evidence(i), '%.2f');
end
% Create the table with the model description and log evidence
T = table(model_desc(:), log_evidence_non_sci(:), ...
    'VariableNames', {'Model', 'Log Evidence'});

% Display the table
disp(T);

%% Run cbm_hbi for top models

% Rank the models by log evidence and get the indices of the top 5
[~, top5_indices] = sort(log_evidence, 'descend');
top5_indices = top5_indices(1:5);

filtered_data = data(valid_subj_all);
filtered_name = cell(1,5);
for i = 1:5
    % Load the saved output for this model
    loaded_data = load(fname{top5_indices(i)});

    loaded_data.cbm.profile.optim.flag = ...
        loaded_data.cbm.profile.optim.flag(valid_subj_all);
    loaded_data.cbm.profile.optim.gradient = ...
        loaded_data.cbm.profile.optim.gradient(:, valid_subj_all);
    loaded_data.cbm.math.A = ...
        loaded_data.cbm.math.A(valid_subj_all);
    loaded_data.cbm.math.Ainvdiag = ...
        loaded_data.cbm.math.Ainvdiag(valid_subj_all);
    loaded_data.cbm.math.lme = ...
        loaded_data.cbm.math.lme(valid_subj_all);
    loaded_data.cbm.math.logdetA = ...
        loaded_data.cbm.math.logdetA(valid_subj_all);
    loaded_data.cbm.math.loglik = ...
        loaded_data.cbm.math.loglik(valid_subj_all);
    loaded_data.cbm.math.theta = ...
        loaded_data.cbm.math.theta(valid_subj_all);
    loaded_data.cbm.output.log_evidence = ...
        loaded_data.cbm.output.log_evidence(valid_subj_all);
    loaded_data.cbm.output.parameters = ...
        loaded_data.cbm.output.parameters(valid_subj_all, :);

    % Save the modified loaded_data back to the same file
    filtered_name{i} = sprintf('ac_refine/filtered_ac_%d.mat', top10_indices(top5_indices(i)));
    save(filtered_name{i}, '-struct', 'loaded_data');
end
fname_hbi = 'hbi_AC5_refined.mat';
cbm_hbi(filtered_data, models(top5_indices), filtered_name, fname_hbi);
cbm_hbi_null(filtered_data, fname_hbi);

% Load the HBI results and display them
fname_hbi_loaded = load(fname_hbi);
hbi_results = fname_hbi_loaded.cbm;
hbi_results.output

hbi_results.input.models
model_names = {'R: 1, S: 3', ...
    'R: 1, S: 2, 3, 4', ...
    'R: 4, S: 1', ...
    'R: 4, S: 2, 3', ...
    'R: 3, S: 1, 2'};
param_names = {'\alpha_W','\alpha_\theta','\gamma', ...
    '\tau', '\theta_0', 'W_0', 'cost'};
% note the latex format
% transformation functions associated with each parameter
transform = {'sigmoid','sigmoid','sigmoid', ...
    'exp', 'none', 'none', 'exp'};

cbm_hbi_plot(fname_hbi, model_names, param_names, transform);
% this function creates a model comparison plot
% (exceedance probability and model frequency)
% plot of transformed parameters of the most frequent model


%% Super refine

models = cell(1,5);
fname = cell(1,5);
for i = 1:5
    models{i} = str2func(sprintf('wrapper_function_%d', top10_indices(top5_indices(i))));
    fname{i} = sprintf('ac_refine/filtered_ac_%d.mat', top10_indices(top5_indices(i)));
end

loaded_data = load('hbi_AC5_refined.mat');
[~, top3_indices] = sort(loaded_data.cbm.output.model_frequency, 'descend');
top3_indices = top3_indices(1:3);
% Create the prior structure for your new model
v = 0;
num_parameters = 9;
% Create the prior structure from previous estimation
prior_ac = struct('mean', zeros(num_parameters, 1), 'variance', v);
top_models = cell(1,3);
top_fname = cell(1,3);
for i = 1:3
    top_models{i} = str2func(sprintf('wrapper_function_%d', top10_indices(top5_indices(top3_indices(i)))));
    top_fname{i} = sprintf('ac_refine/refine_ac_%d.mat', top10_indices(top5_indices(top3_indices(i))));
    prior_ac.mean = prior_ac.mean + ...
        loaded_data.cbm.output.group_mean{top3_indices(i)}';
    prior_ac.variance = max(prior_ac.variance, ...
        max(loaded_data.cbm.output.group_hierarchical_errorbar{top3_indices(1)}));
    top_fname{i} = sprintf('ac_refine/top3_ac_%d.mat', top10_indices(top5_indices(top3_indices(i))));
end
prior_ac.mean = prior_ac.mean/3;
prior_ac.variance = min(6.25, ...
    (prior_ac.variance)^2 * length(filtered_data));

% Initialize a parallel pool if it doesn't already exist
if isempty(gcp('nocreate'))
    parpool;
end
% Populate the top 3 models and their corresponding fcbm_maps
pconfig = struct();
pconfig.numinit = min(70*num_parameters, 1000);
pconfig.numinit_med = 1000;
pconfig.numinit_up = 10000;
pconfig.tolgrad = 1e-4;
pconfig.tolgrad_liberal = 0.01;
parfor i = 1:3
    % Run the cbm_lap function for your new model
    cbm_lap(data, top_models{i}, prior_ac, top_fname{i},pconfig);
end

% Filter out invalid subjects
valid_subj_all = ones(1,210);
for i = 1:3
    % Load the saved output for this model
    loaded_data = load(top_fname{i});
    
    % Initialize a logical index for valid subjects
    valid_subjects = ~isnan(loaded_data.cbm.math.logdetA) ...
        & ~isinf(loaded_data.cbm.math.logdetA) ...
        & (loaded_data.cbm.math.logdetA ~= 0);
    % Calculate the mean and standard deviation of logdetA for valid subjects
    mean_logdetA = mean(loaded_data.cbm.math.logdetA(valid_subjects));
    std_logdetA = std(loaded_data.cbm.math.logdetA(valid_subjects));
    
    % Add the condition for logdetA values within 3 standard deviations of the mean
    valid_subjects = valid_subjects & ...
        (abs(loaded_data.cbm.math.logdetA - mean_logdetA) <= 3 * std_logdetA);

    valid_subj_all = valid_subj_all & valid_subjects;
end
filtered_data = data(valid_subj_all);
filtered_name = cell(1,3);
for i = 1:3
    % Load the saved output for this model
    loaded_data = load(top_fname{i});

    loaded_data.cbm.profile.optim.flag = ...
        loaded_data.cbm.profile.optim.flag(valid_subj_all);
    loaded_data.cbm.profile.optim.gradient = ...
        loaded_data.cbm.profile.optim.gradient(:, valid_subj_all);
    loaded_data.cbm.math.A = ...
        loaded_data.cbm.math.A(valid_subj_all);
    loaded_data.cbm.math.Ainvdiag = ...
        loaded_data.cbm.math.Ainvdiag(valid_subj_all);
    loaded_data.cbm.math.lme = ...
        loaded_data.cbm.math.lme(valid_subj_all);
    loaded_data.cbm.math.logdetA = ...
        loaded_data.cbm.math.logdetA(valid_subj_all);
    loaded_data.cbm.math.loglik = ...
        loaded_data.cbm.math.loglik(valid_subj_all);
    loaded_data.cbm.math.theta = ...
        loaded_data.cbm.math.theta(valid_subj_all);
    loaded_data.cbm.output.log_evidence = ...
        loaded_data.cbm.output.log_evidence(valid_subj_all);
    loaded_data.cbm.output.parameters = ...
        loaded_data.cbm.output.parameters(valid_subj_all, :);

    % Save the modified loaded_data back to the same file
    filtered_name{i} = sprintf('ac_refine/filtered_ac_%d.mat', top10_indices(top5_indices(top3_indices(i))));
    save(filtered_name{i}, '-struct', 'loaded_data');
end
fname_hbi = 'hbi_AC3_refined.mat';
cbm_hbi(filtered_data, top_models, filtered_name, fname_hbi);

% Load the HBI results and store them
fname_hbi_loaded = load(fname_hbi);
hbi_results = fname_hbi_loaded.cbm;
hbi_results.output

model_names = T.Model(top5_indices(top3_indices));
param_names = {'\alpha_W','\alpha_\theta','\gamma', ...
    '\tau', '\theta_0', 'W_0', ...
    'c_1', 'c_2', 'c_3'};   
% note the latex format
% transformation functions associated with each parameter
transform = {'sigmoid','sigmoid','sigmoid', ...
    'exp', 'none', 'none', ...
    'exp','exp','exp'};

cbm_hbi_plot(fname_hbi, model_names, param_names, transform);
