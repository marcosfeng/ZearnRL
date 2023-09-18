
% load data
fdata = load('../data/all_data.mat');
data  = fdata.data;

% Define prior
prior.mean = [0.5; 0.5; 0.5; ones(C, 1)];
prior.variance = eye(length(prior.mean));

% Add path to CBM toolbox
addpath(fullfile('..' ,'codes'));

% Run model fitting
output_file = 'q_learning_fit.mat';
cbm_lap(data_cell, @q_learning_model, prior, output_file);

% Load and display results
load('q_learning_fit.mat');
disp(['MAP estimates: ', num2str(map)]);
disp(['Log evidence: ', num2str(log_evidence)]);
