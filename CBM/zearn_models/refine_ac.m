% Add path to the codes directory
addpath(fullfile('..','codes'));

% Add path to the wrappers directory
addpath(fullfile('..','zearn_models','wrappers'));

% Create the prior structure for your new model
v = 2;
num_parameters = 7;
prior_ac = struct('mean', zeros(num_parameters, 1), 'variance', v);

% Load the common data for all datasets
fdata = load('../data/all_data.mat');
data  = fdata.data;

models = {@wrapper_function_9,@wrapper_function_12, ...
    @wrapper_function_13,@wrapper_function_14};

fcbm_maps = {'lap_ac_9.mat','lap_ac_12.mat', ...
    'lap_ac_13.mat','lap_ac_14.mat'};

% Run cbm_hbi for all models
fname_hbi = 'hbi_AC_refined.mat';
cbm_hbi(data, models, fcbm_maps, fname_hbi);

% Load the HBI results and store them
fname_hbi_loaded = load(fname_hbi);
hbi_results = fname_hbi_loaded.cbm;

% Save the HBI results for all datasets
save('hbi_AC_results.mat', 'hbi_results');