% Add path to the codes directory
addpath(fullfile('..','codes'));

% Add path to the wrappers directory
addpath(fullfile('..','zearn_models','wrappers'));

% Define the prior variance
v = 2;

% Determine the number of parameters in your model.
num_parameters = 7;

% Create the prior structure for your new model
prior_ac = struct('mean', zeros(num_parameters, 1) - 0.7, 'variance', v);
prior_ac.mean(3) = prior_ac.mean(3) + 3;

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

% Run cbm_hbi for all models
fname_hbi = 'hbi_AC.mat';
cbm_hbi(data, models, fcbm_maps, fname_hbi);

% Load the HBI results and store them
fname_hbi_loaded = load(fname_hbi);
hbi_results = fname_hbi_loaded.cbm;

% Save the HBI results for all datasets
save('hbi_AC_results.mat', 'hbi_results');