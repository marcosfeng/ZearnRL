rng(37909890)
% © 1998-2023 RANDOM.ORG
% Timestamp: 2023-10-04 18:38:28 UTC

% Add path to the codes directory
addpath(fullfile('..','codes'));
addpath(fullfile('..','zearn_codes'));
addpath(fullfile('..','zearn_codes','ac_wrappers'));

% Load the common data for all datasets
fdata = load('../data/sample_data.mat');
data  = fdata.data;

% Initialize models and fcbm_maps arrays
numFiles = numel(dir(fullfile('..', 'zearn_codes', 'ac_wrappers')));
numFiles = numFiles - 2;
models = cell(1, numFiles);
fcbm_maps = cell(1, numFiles);

% Initialize a parallel pool if it doesn't already exist
if isempty(gcp('nocreate'))
    parpool;
end
% Define the prior variance
v = 6.25;
% Loop through each of the wrapper functions
parfor wrapper_num = 1:numFiles
    % Determine the number of parameters in your model.
    % Read the contents of the wrapper function file
    file_contents = fileread( ...
        sprintf('wrapper_function_%d.m', wrapper_num));
    % Find the subj.state assignment line
    state_line = regexp(file_contents, ...
        'subj\.state\s*=\s*\[.*?\]', 'match', 'once');
    % Extract the elements inside the square brackets
    elements = regexp(state_line, 'subj\.\w+', 'match');
    % Count the number of parameters
    num_states = numel(elements);
    num_parameters = 5 + 2*num_states;

    % Create the prior structure for your new model
    prior_ac = struct('mean', zeros(num_parameters, 1), 'variance', v);
    % Create the PCONFIG struct
    pconfig = struct();
    pconfig.numinit = min(70*num_parameters, 1000);
    pconfig.numinit_med = 1000;
    pconfig.numinit_up = 10000;
    pconfig.tolgrad = 1e-4;
    pconfig.tolgrad_liberal = 0.01;

    % Specify the file-address for saving the output
    fname = sprintf('ac_subj_results/lap_ac_%d.mat', wrapper_num);
    
    % Get the function handle for the current wrapper function
    wrapper_func = str2func(sprintf('wrapper_function_%d', wrapper_num));
    
    % Run the cbm_lap function for your new model
    cbm_lap(data, wrapper_func, prior_ac, fname, pconfig);

    % Store the function handle and file name for later
    models{wrapper_num} = wrapper_func;
    fcbm_maps{wrapper_num} = fname;
end

%% 

% Add path to the codes directory
addpath(fullfile('ac_subj_results'));

% Initialize an array to store the log evidence for each model
log_evidence = zeros(1, numFiles);
valid_subj_cell = cell(1,numFiles);

% Loop through each of the 75 saved files to extract log evidence
for wrapper_num = 1:numFiles
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
histogram(log_evidence, 15);
title('Histogram of Log Evidences');
xlabel('Log Evidence');
ylabel('Frequency');

% Rank the models by log evidence and get the indices of the top
[~, top10_indices] = sort(log_evidence, 'descend');
top10_indices = top10_indices(1:10);

% Save the new data file
save('top10_ac.mat', 'top10_indices');
