% Load data
load('data/stan_data.mat');

% Prepare data structure
data = struct('N', N, 'Tsubj', Tsubj, 'choice', choice, 'outcome', outcome, 'state', state);

data_cell = cell(1, N);  % Initialize cell array
for i = 1:N
    data_cell{i} = struct('Tsubj', Tsubj(i), 'choice', choice(i, :, :), 'outcome', outcome(i, :), 'state', state(i, :));
end

% Define prior
prior.mean = [0.5; 0.5; 0.5; ones(C, 1)];
prior.variance = eye(length(prior.mean));

% Add path to CBM toolbox
addpath(fullfile('codes'));

% Run model fitting
output_file = 'q_learning_fit.mat';
cbm_lap(data_cell, @q_learning_model, prior, output_file);

% Load and display results
load('q_learning_fit.mat');
disp(['MAP estimates: ', num2str(map)]);
disp(['Log evidence: ', num2str(log_evidence)]);
