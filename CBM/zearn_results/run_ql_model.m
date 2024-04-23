rng(37909890)
% © 1998-2023 RANDOM.ORG
% Timestamp: 2023-10-04 18:38:28 UTC

% Add path to the codes directory
addpath(fullfile('..','codes'));
addpath(fullfile('..','zearn_codes'));
addpath(fullfile('..','zearn_codes','ql_wrappers'));

% Load data
fdata = load('../data/sample_data.mat');
data  = fdata.data;

% Initialize models and fcbm_maps arrays
Files = dir(fullfile('..', 'zearn_codes', 'ql_wrappers'));
numFiles = sum(startsWith({Files.name}, 'wrapper_'));
models = cell(1, numFiles);
fcbm_maps = cell(1, numFiles);

% Define the prior variance
v = 2;

% Determine the number of parameters in your model.
num_parameters = 4;

% Create the prior structure for your model
prior_ql = struct('mean', zeros(num_parameters, 1), 'variance', v);

if isempty(gcp('nocreate'))
    parpool;
end
% Loop through each of the wrapper functions
parfor wrapper_num = 1:numFiles
    % Specify the file-address for saving the output
    fname = sprintf('ql_subj_results/lap_ql_%d.mat', wrapper_num);
    
    % Get the function handle for the current wrapper function
    wrapper_func = str2func(sprintf('wrapper_function_%d', wrapper_num));
    
    % Run the cbm_lap function for your new model
    cbm_lap(data, wrapper_func, prior_ql, fname);

    % Store the function handle and file name for later
    models{wrapper_num} = wrapper_func;
    fcbm_maps{wrapper_num} = fname;
end

% Add path to the codes directory
addpath(fullfile('ql_subj_results'));

% Initialize an array to store the log evidence for each model
log_evidence = zeros(1, numFiles);
valid_subj_cell = cell(1,numFiles);

% Loop through each of the saved files to extract log evidence
for wrapper_num = 1:numFiles
    % Load the saved output for this model
    fname = sprintf('lap_ql_%d.mat', wrapper_num);
    loaded_data = load(fname);
    
    % Initialize a logical index for valid subjects
    valid_subjects = ~isnan(loaded_data.cbm.math.logdetA) ...
        & ~isinf(loaded_data.cbm.math.logdetA) ...
        & (loaded_data.cbm.math.logdetA ~= 0);
    % Calculate the mean and standard deviation of logdetA for valid subjects
    mean_logdetA = mean(loaded_data.cbm.math.logdetA(valid_subjects));
    std_logdetA = std(loaded_data.cbm.math.logdetA(valid_subjects));
    
    % Add the condition for logdetA values within 3 standard deviations of the mean
    valid_subjects = valid_subjects & ...
        (abs(loaded_data.cbm.math.logdetA - mean_logdetA) <= 2 * std_logdetA);

    % Filter out invalid subjects from the log-likelihood array
    log_evidence(wrapper_num) = ...
        sum(loaded_data.cbm.output.log_evidence(valid_subjects));
    valid_subj_cell{wrapper_num} = valid_subjects;
end

% Create a histogram of the log evidences
figure;
histogram(log_evidence);
title('Histogram of Log Evidences');
xlabel('Log Evidence');
ylabel('Frequency');

% Rank the models by log evidence
[~, ranked_indices] = sort(log_evidence, 'descend');
ranked_indices = ranked_indices(1:4);

% Save the new data file
save('ranked_ql.mat', 'ranked_indices');
