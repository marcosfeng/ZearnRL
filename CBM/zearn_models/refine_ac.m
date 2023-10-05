% Add paths
addpath(fullfile('..','codes'));
addpath(fullfile('..','zearn_models','wrappers'));
addpath(fullfile('ac_subj_results'));

% Load the common data for all datasets
load("top5_indeces.mat");
fdata = load('../data/all_data.mat');
data  = fdata.data;

% Initialize a parallel pool if it doesn't already exist
if isempty(gcp('nocreate'))
    parpool;
end

models = cell(1,5);
fname = cell(1,5);

% Create the prior structure for your new model
v = 6.25;
num_parameters = 9;
% Create the prior structure for your new model
prior_ac = struct('mean', zeros(num_parameters, 1), 'variance', v);
for i = 1:5
    models{i} = str2func(sprintf('wrapper_function_%d', top5_indices(i)));
    fname{i} = sprintf('ac_refine/refine_ac_%d.mat', top5_indices(i));

    loaded_data = load(fname{i});
    prior_ac.mean = prior_ac.mean + ...
        mean(loaded_data.cbm.output.parameters,1)';
end

prior_ac.mean = prior_ac.mean/5;

% Populate the top 5 models and their corresponding fcbm_maps
parfor i = 1:5
    % Run the cbm_lap function for your new model
    cbm_lap(data, models{i}, prior_ac, fname{i});
end

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

% Filter out invalid subjects from the original data
filtered_data = data(valid_subj_all);
parfor i = 1:5
    % Run the cbm_lap function for your new model
    cbm_lap(filtered_data, models{i}, prior_ac, fname{i});
end

% Run cbm_hbi for all models
fname_hbi = 'hbi_AC_refined.mat';
cbm_hbi(data, models, fname, fname_hbi);

% Load the HBI results and store them
fname_hbi_loaded = load(fname_hbi);
hbi_results = fname_hbi_loaded.cbm;
hbi_results.output

model_names = {'Ba, M', 'A, Ba, M', 'A, Bo, M', 'Ba, Bo, M'};
param_names = {'\alpha_w','\alpha_\theta','\gamma', ...
    '\tau', 'c_1', 'c_2', 'c_3'};
% note the latex format
% transformation functions associated with each parameter
transform = {'exp','exp','sigmoid','exp','exp','exp','exp'};

cbm_hbi_plot(fname_hbi, model_names, param_names, transform)
% this function creates a model comparison plot
% (exceednace probability and model frequency)
% plot of transformed parameters of the most frequent model
