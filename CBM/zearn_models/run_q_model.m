addpath(fullfile('..','codes'));

% load data
fdata = load('../data/all_data.mat');
data  = fdata.data;

% Define the prior variance
v = 2;

% Determine the number of parameters in your model.
num_parameters = 6;

% Create the prior structure for your model
prior_ql = struct('mean', zeros(num_parameters, 1), 'variance', v);

% Specify the file-address for saving the output
fname = 'lap_ql.mat';  % Laplace results for Q-learning model

% Assuming `data` is defined and accessible
% Run the cbm_lap function for your Q-learning model
cbm_lap(data, @q_learning_model, prior_ql, fname);

% Load and display results
models = {@q_learning_model};
fcbm_maps = {'lap_ql.mat'};
fname_hbi = 'hbi_lap_ql.mat';

cbm_hbi(data,models,fcbm_maps,fname_hbi);

fname_hbi  = load("hbi_lap_ql.mat");
cbm = fname_hbi.cbm;
cbm.output
