rng(37909890)
% © 1998-2023 RANDOM.ORG
% Timestamp: 2023-10-04 18:38:28 UTC

% Add paths
addpath(fullfile('..','codes'));
addpath(fullfile('..', 'zearn_codes'));
addpath(fullfile('..', 'zearn_codes', 'ql_wrappers'));
addpath(fullfile('ql_subj_results'));

% Load prechosen models and common data
load("ranked_ql.mat");
fdata = load('../data/all_data.mat');
data = fdata.data;

%% Re-estimate top models with more precision

models = cell(1, length(ranked_indices));
fname = cell(1, length(ranked_indices));

% Update the prior structure for Q-learning models
v = 1;
num_parameters = 4;
prior_ql = struct('mean', zeros(num_parameters, 1), 'variance', v);

for i = 1:length(ranked_indices)
    index = ranked_indices(i); % Get the model index
    models{i} = str2func(sprintf('wrapper_function_%d', index));
    fname{i} = sprintf('ql_refine/refine_ql_%d.mat', index);

    % Load previously saved results for mean updates
    loaded_data = load(sprintf('ql_subj_results/lap_ql_%d.mat', index));
    prior_ql.mean = prior_ql.mean + ...
        mean(loaded_data.cbm.output.parameters, 1)';
end
prior_ql.mean = prior_ql.mean / length(ranked_indices);

% PCONFIG structure with refined setup for Q-learning models
pconfig = struct();
pconfig.numinit = min(140 * num_parameters, 2000);
pconfig.numinit_med = 2000;
pconfig.numinit_up = 20000;
pconfig.tolgrad = 5e-5;
pconfig.tolgrad_liberal = 0.005;

% Ensure a parallel pool is available
if isempty(gcp('nocreate'))
    parpool;
end
% Refined estimation loop
parfor i = 1:length(ranked_indices)
    cbm_lap(data, models{i}, prior_ql, fname{i}, pconfig);
end

%% Histograms by valid log evidence

model_desc = cell(size(fname));
valid_subj_all = ones(1,295);
% Loop over each file name to construct the model description
for i = 1:length(fname)
    loaded_data = load(fname{i});

    % Create a logical index for valid subjects
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
    % Create a cell array for the model descriptions
    % Extract the number from the filename
    num = regexp(fname{i}, '\d+', 'match');
    num = num{1}; % Assuming there's only one number in the filename
    
    % Construct the path to the corresponding .m file
    wrapper_filename = fullfile('..', 'zearn_codes', ...
        'ql_wrappers', ['wrapper_function_' num '.m']);
    
    % Read the .m file
    try
        file_contents = fileread(wrapper_filename);
    catch
        model_desc{i} = sprintf('Model %s: File not found', num);
        continue; % Skip this iteration if file not found
    end

    % Extract the outcome variable name
    outcome_match = regexp(file_contents, ...
        'subj\.outcome\s*=\s*subj\.(\w+);', 'tokens');
    % Extract the outcome variable name
    action_match  = regexp(file_contents, ...
        'subj\.action\s*=\s*subj\.(\w+);', 'tokens');

    if ~isempty(outcome_match) && ~isempty(action_match)
        outcome_var = outcome_match{1}{1};
        action_var  = action_match{1}{1};
        
        % Construct the model description
        model_desc{i} = sprintf('Action: %s, Outcome: %s', ...
            (action_var), (outcome_var));
    else
        % If the pattern is not found, use a placeholder
        model_desc{i} = 'Outcome and Actions not found in wrapper';
    end
end

% Convert the log evidence to non-scientific notation using a cell array
log_evidence = zeros(1, length(fname));
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

% Rank the models by log evidence and get the indices of the top 4
[~, top4_indices] = sort(log_evidence, 'descend');

filtered_data = data(valid_subj_all);
filtered_name = cell(1, 4);
for i = 1:4
    % Load the saved output for this model
    loaded_data = load(fname{top4_indices(i)});

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
    filtered_name{i} = sprintf('ql_refine/filtered_ql_%d.mat', ...
        ranked_indices(top4_indices(i)));
    save(filtered_name{i}, '-struct', 'loaded_data');
end

fname_hbi = 'hbi_QL4_refined.mat';
cbm_hbi(filtered_data, models(top4_indices), filtered_name, fname_hbi);
cbm_hbi_null(filtered_data, fname_hbi);

% Load the HBI results and display them
fname_hbi_loaded = load(fname_hbi);
hbi_results = fname_hbi_loaded.cbm;
hbi_results.output

hbi_results.input.models

model_names = {'R: 4', ...
    'R: 2', ...
    'R: 1', ...
    'R: 3'};

% Define parameter names specific to Q-learning models
param_names = {'\alpha', '\gamma', '\tau', 'C'};

% Define transformation functions for parameters
transform = {'sigmoid', 'sigmoid', 'exp', 'exp'};

% Plot HBI analysis results
cbm_hbi_plot(fname_hbi, model_names, param_names, transform);
