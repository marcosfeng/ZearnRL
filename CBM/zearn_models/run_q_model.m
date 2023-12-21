addpath(fullfile('..','codes'));

% load data
fdata = load('../data/all_data.mat');
data  = fdata.data;

% Initialize an empty array to store alerts values
allAlerts = [];
% Loop through each cell in the cell array
for i = 1:size(data, 1)
    % Append the alerts values from the current cell to the array
    allAlerts = [allAlerts; data{i}.alerts];
end
% Calculate the median of all alerts values
medianValue = median(allAlerts);
for i = 1:size(data, 1)
    % Append the alerts values from the current cell to the array
    data{i}.medianAlerts = medianValue;
end

% Define the prior variance
v = 2;

% Determine the number of parameters in your model.
num_parameters = 6;

% Create the prior structure for your model
prior_ql = struct('mean', zeros(num_parameters, 1), 'variance', v);

% Specify the file-address for saving the output
fname = {'lap_ql.mat','lap_ql_state.mat'};  % Laplace results for Q-learning model

% Run the cbm_lap function for your Q-learning models
cbm_lap(data, @q_model, prior_ql, fname{1});
cbm_lap(data, @q_state_model, prior_ql, fname{2});

% Load and display results
models = {@q_model, @q_state_model};
fname_hbi = 'hbi_lap_ql.mat';

cbm_hbi(data,models,fname,fname_hbi);

fname_hbi  = load("hbi_lap_ql.mat");
cbm = fname_hbi.cbm;
cbm.output
