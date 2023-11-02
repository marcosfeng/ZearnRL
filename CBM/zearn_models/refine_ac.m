rng(37909890)
% © 1998-2023 RANDOM.ORG
% Timestamp: 2023-10-04 18:38:28 UTC

%% Re-estimate top 5 models with more precision

% Add paths
addpath(fullfile('..','codes'));
addpath(fullfile('..','zearn_models','wrappers'));
addpath(fullfile('ac_subj_results'));

% Load the common data for all datasets
load("top5_indeces.mat");
fdata = load('../data/all_data.mat');
data  = fdata.data;
models = cell(1,5);
fname = cell(1,5);

% Create the prior structure for your new model
v = 6.25;
num_parameters = 9;
% Create the prior structure from previous estimation
prior_ac = struct('mean', zeros(num_parameters, 1), 'variance', v);
for i = 1:5
    models{i} = str2func(sprintf('wrapper_function_%d', top5_indices(i)));
    fname{i} = sprintf('ac_subj_results/lap_ac_%d.mat', top5_indices(i));

    loaded_data = load(fname{i});
    prior_ac.mean = prior_ac.mean + ...
        mean(loaded_data.cbm.output.parameters,1)';
    fname{i} = sprintf('ac_refine/refine_ac_%d.mat', top5_indices(i));
end
prior_ac.mean = prior_ac.mean/5;

% Populate the top 5 models and their corresponding fcbm_maps
% Create the PCONFIG struct
pconfig = struct();
pconfig.numinit = min(50*num_parameters, 500);
pconfig.numinit_med = 500;
pconfig.numinit_up = 5000;
pconfig.tolgrad = 2e-4;
pconfig.tolgrad_liberal = 0.02;
pconfig.range = [-10*ones(1,num_parameters); 10*ones(1,num_parameters)];
% Initialize a parallel pool if it doesn't already exist
if isempty(gcp('nocreate'))
    parpool;
end
parfor i = 1:5
    % Run the cbm_lap function for your new model
    cbm_lap(data, models{i}, prior_ac, fname{i},pconfig);
end

%% Histograms by valid log evidence

valid_subj_all = ones(1,210);
for i = 1:5
    % Load the saved output for this model
    loaded_data = load(fname{i});
    
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

log_evidence = zeros(1, 5);
for i = 1:5
    loaded_data = load(fname{i});
    log_evidence(i) = ...
        sum(loaded_data.cbm.output.log_evidence(valid_subj_all));
end

% Create a histogram of the log evidences
figure;
histogram(log_evidence,100);
title('Histogram for Top 5 Models');
xlabel('Log Evidence');
ylabel('Frequency');

%% Run cbm_hbi for top models

% Filter out invalid subjects from the original data
valid_subj_all = ones(1,210);
for i = 1:5
    % Load the saved output for this model
    loaded_data = load(fname{i});
    
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
filtered_name = cell(1,5);
for i = 1:5
    % Load the saved output for this model
    loaded_data = load(fname{i});

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
    filtered_name{i} = sprintf('ac_refine/filtered_ac_%d.mat', top5_indices(i));
    save(filtered_name{i}, '-struct', 'loaded_data');
end
fname_hbi = 'hbi_AC5_refined.mat';
cbm_hbi(filtered_data, models, filtered_name, fname_hbi);

% Load the HBI results and store them
fname_hbi_loaded = load(fname_hbi);
hbi_results = fname_hbi_loaded.cbm;
hbi_results.output

model_names = {'Ba ~ M', ...
    'Ba ~ A', ...
    'M ~ A, Bo', ...
    'Bo ~ ASt, A, Ba', ...
    'Bo ~ A'};
param_names = {'\alpha_W','\alpha_\theta','\gamma', ...
    '\tau', '\theta_0', 'W_0', ...
    'c_1', 'c_2', 'c_3'};
% note the latex format
% transformation functions associated with each parameter
transform = {'sigmoid','sigmoid','sigmoid', ...
    'exp', 'none', 'none', ...
    'exp','exp','exp'};

cbm_hbi_plot(fname_hbi, model_names, param_names, transform);
% this function creates a model comparison plot
% (exceednace probability and model frequency)
% plot of transformed parameters of the most frequent model
