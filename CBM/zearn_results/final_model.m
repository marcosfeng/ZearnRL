rng(37909890)
% © 1998-2023 RANDOM.ORG
% Timestamp: 2023-10-04 18:38:28 UTC

% Add path to the codes directory
addpath(fullfile('..','codes'));
addpath(fullfile('..','zearn_codes'));
addpath(fullfile('..','zearn_codes','ac_wrappers'));

% Load the common data for all datasets
fdata = load('../data/full_data.mat');
data  = fdata.data;

num_parameters = 7;
% Define the prior variance
v = 6.25;
num_subjects = length(data);
priors = struct('mean', zeros(num_parameters, 1), 'variance', v);

parfor i = 1:num_subjects
    % Construct filename for saving output
    fname = sprintf('top_results/lap_ac2_%d.mat', i);
    % Run the cbm_lap function for the current model and subject
    cbm_lap(data(i), @wrapper_function_2, priors, fname);
end
