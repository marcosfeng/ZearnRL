% Add path to the codes directory
addpath(fullfile('..','codes'));

% Add path to the wrappers directory
addpath(fullfile('..','zearn_models','wrappers'));

% Create the prior structure for your new model
v = 6.25;
num_parameters = 7;
prior_ac = struct('mean', zeros(num_parameters, 1), 'variance', v);

% Load the common data for all datasets
fdata = load('../data/all_data.mat');
data  = fdata.data;

% Initialize a parallel pool if it doesn't already exist
if isempty(gcp('nocreate'))
    parpool;
end

models = {@wrapper_function_9,@wrapper_function_12, ...
    @wrapper_function_13,@wrapper_function_14};

fcbm_maps = {'lap_ac_9.mat','lap_ac_12.mat', ...
    'lap_ac_13.mat','lap_ac_14.mat'};

parfor i = 1:4
    % Specify the file-address for saving the output
    fname = fcbm_maps{i};
    
    % Run the cbm_lap function for your new model
    cbm_lap(data, models{i}, prior_ac, fname);
end

% Run cbm_hbi for all models
fname_hbi = 'hbi_AC_refined.mat';
cbm_hbi(data, models, fcbm_maps, fname_hbi);

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
