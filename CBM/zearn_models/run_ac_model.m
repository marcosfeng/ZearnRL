rng(37909890)
% © 1998-2023 RANDOM.ORG
% 37909890	
% Timestamp: 2023-10-04 18:38:28 UTC

% Add path to the codes directory
addpath(fullfile('..','codes'));

% Add path to the wrappers directory
addpath(fullfile('..','zearn_models','wrappers'));

% Define the prior variance
v = 2;

% Determine the number of parameters in your model.
num_parameters = 9;

% Create the prior structure for your new model
prior_ac = struct('mean', zeros(num_parameters, 1) - 0.7, 'variance', v);
prior_ac.mean(3) = prior_ac.mean(3) + 3;
prior_ac.mean(5:end) = 0;

% Load the common data for all datasets
fdata = load('../data/all_data.mat');
data  = fdata.data;

% Initialize models and fcbm_maps arrays
models = cell(1, 75);
fcbm_maps = cell(1, 75);

% Initialize a parallel pool if it doesn't already exist
if isempty(gcp('nocreate'))
    parpool;
end

% Loop through each of the 75 wrapper functions
parfor wrapper_num = 1:75
    % Specify the file-address for saving the output
    fname = sprintf('lap_ac_%d.mat', wrapper_num);  % Laplace results for actor-critic model
    
    % Get the function handle for the current wrapper function
    wrapper_func = str2func(sprintf('wrapper_function_%d', wrapper_num));
    
    % Run the cbm_lap function for your new model
    cbm_lap(data, wrapper_func, prior_ac, fname);

    % Store the function handle and file name for later
    models{wrapper_num} = wrapper_func;
    fcbm_maps{wrapper_num} = sprintf('lap_ac_%d.mat', wrapper_num);
end

% Initialize an array to store the log evidence for each model
log_evidence = zeros(1, 75);

% Loop through each of the 75 saved files to extract log evidence
for wrapper_num = 1:75
    % Load the saved output for this model
    fname = sprintf('lap_ac_%d.mat', wrapper_num);
    loaded_data = load(fname);
    
    % Extract and store the log evidence for this model
    log_evidence(wrapper_num) = sum(loaded_data.cbm.output.log_evidence);
end

% Create a histogram of the log evidences
figure;
histogram(log_evidence, 20); % 20 bins for illustration, you can adjust this
title('Histogram of Log Evidences');
xlabel('Log Evidence');
ylabel('Frequency');

% Rank the models by log evidence and get the indices of the top 5
[~, top5_indices] = sort(log_evidence, 'descend');
top5_indices = top5_indices(1:5);

% Initialize models and fcbm_maps arrays for the top 5 models
top5_models = cell(1, 5);
top5_fcbm_maps = cell(1, 5);

% Populate the top 5 models and their corresponding fcbm_maps
for i = 1:5
    top5_models{i} = str2func(sprintf('wrapper_function_%d', top5_indices(i)));
    top5_fcbm_maps{i} = sprintf('lap_ac_%d.mat', top5_indices(i));
end

% Load the common data for all datasets
fdata = load('../data/all_data.mat');
data  = fdata.data;

% Run cbm_hbi for the top 5 models
fname_hbi = 'hbi_AC_top5.mat';
cbm_hbi(data, top5_models, top5_fcbm_maps, fname_hbi);

% Load the HBI results and store them
fname_hbi_loaded = load(fname_hbi);
hbi_results = fname_hbi_loaded.cbm;

% Save the HBI results for the top 5 models
save('AC_hbi_top5_results.mat', 'hbi_results');




% Run cbm_hbi for all models
fname_hbi = 'hbi_AC.mat';
cbm_hbi(data, models, fcbm_maps, fname_hbi);

% Load the HBI results and store them
fname_hbi_loaded = load(fname_hbi);
hbi_results = fname_hbi_loaded.cbm;

% Save the HBI results for all datasets
save('hbi_AC_results.mat', 'hbi_results');