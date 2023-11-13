rng(37909890)
% © 1998-2023 RANDOM.ORG
% Timestamp: 2023-10-04 18:38:28 UTC

% Add path to the codes directory
addpath(fullfile('..','codes'));

% Add path to the wrappers directory
addpath(fullfile('..','zearn_models','wrappers'));

% Define the prior variance
v = 6.25;
% Determine the number of parameters in your model.
num_parameters = 3*5;
% Create the prior structure for your new model
prior_logit = struct('mean', zeros(num_parameters, 1), 'variance', v);

% Load the common data for all datasets
fdata = load('../data/all_data.mat');
data  = fdata.data;

% Initialize models and fcbm_maps arrays
models = cell(1, 75);
fcbm_maps = cell(1, 75);

% Specify the file-address for saving the output
fname = 'logit_subj_results/lap_logit.mat';
% Run the cbm_lap function for your new model
cbm_lap(data, @logit_model, prior_logit, fname);

%% 

% Add path to the codes directory
addpath(fullfile('ac_subj_results'));

% Initialize an array to store the log evidence for each model
log_evidence = zeros(1, 75);
valid_subj_cell = cell(1,75);

% Loop through each of the 75 saved files to extract log evidence
for wrapper_num = 1:75
    % Load the saved output for this model
    fname = sprintf('lap_ac_%d.mat', wrapper_num);
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

% Rank the models by log evidence and get the indices of the top 5
[~, top10_indices] = sort(log_evidence, 'descend');
top10_indices = top10_indices(1:10);

% Save the new data file
save('top10_indeces.mat', 'top10_indices');
