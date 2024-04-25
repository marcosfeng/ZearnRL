rng(37909890)
% © 1998-2023 RANDOM.ORG
% Timestamp: 2023-10-04 18:38:28 UTC

% Add path to the codes directory
addpath(fullfile('..','codes'));
addpath(fullfile('..','zearn_codes'));
addpath(fullfile('..','zearn_codes','ac_wrappers'));
addpath(fullfile('..','zearn_codes','hybrid_wrappers'));

% Load the common data for all datasets
fdata = load('../data/full_data.mat');
data  = fdata.data;

%% Models

ac_parameters = 7;
% Define the prior variance
v = 6.25;
num_subjects = length(data);
% Create the PCONFIG struct
pconfig = struct();
pconfig.numinit = min(70*ac_parameters, 1000);
pconfig.numinit_med = 1000;
pconfig.numinit_up = 10000;
pconfig.tolgrad = 1e-4;
pconfig.tolgrad_liberal = 0.01;

% Estimate Logit Model
logit_parameters = 8;
priors = struct('mean', zeros(logit_parameters, 1), 'variance', v);
parfor i = 1:num_subjects
    % Construct filename for saving output
    fname = sprintf('top_results/logit/lap_logit7_%d.mat', i);
    % Run the cbm_lap function for the current model and subject
    cbm_lap(data(i), @logit_wrapper_7, priors, fname, pconfig);
end
% Aggregate results for each subject
fname_subjs = cell(num_subjects,1);
for subj = 1:num_subjects
    % Construct the filename for the current subject's results
    fname_subjs{subj} = sprintf('top_results/logit/lap_logit7_%d.mat', subj);
end
cbm_lap_aggregate(fname_subjs,'top_results/lap_logit7.mat');


% Estimate AC Model
priors = struct('mean', zeros(ac_parameters, 1), 'variance', v);
parfor i = 1:num_subjects
    % Construct filename for saving output
    fname = sprintf('top_results/ac/lap_ac15_%d.mat', i);
    % Run the cbm_lap function for the current model and subject
    cbm_lap(data(i), @wrapper_function_15, priors, fname, pconfig);
end
% Aggregate results for each subject
fname_subjs = cell(num_subjects,1);
for subj = 1:num_subjects
    % Construct the filename for the current subject's results
    fname_subjs{subj} = sprintf('top_results/ac/lap_ac15_%d.mat', subj);
end
cbm_lap_aggregate(fname_subjs,'top_results/lap_ac15.mat');

% Check convergence
valid_subj_all = ones(1,num_subjects);
% Loop over each file name to construct the model description
loaded_data = load('top_results/lap_ac15.mat');
% 1) Create a logical index for valid subjects
valid_subjects = ~isnan(loaded_data.cbm.math.logdetA) ...
    & ~isinf(loaded_data.cbm.math.logdetA) ...
    & (loaded_data.cbm.math.logdetA ~= 0) ...
    & imag(loaded_data.cbm.math.loglik) == 0 ...
    & imag(loaded_data.cbm.math.lme) == 0;
% Calculate the mean and SD of logdetA for valid subjects
q25 = quantile(loaded_data.cbm.math.loglik(valid_subjects),0.25);
q75 = quantile(loaded_data.cbm.math.loglik(valid_subjects),0.75);
% Add the condition for logdetA values within 3 SDs of the mean
valid_subjects = valid_subjects & ...
loaded_data.cbm.math.loglik >= q25 - 1.5*(q75-q25) & ...
loaded_data.cbm.math.loglik <= q75 + 1.5*(q75-q25);
valid_subj_all = valid_subj_all & valid_subjects;

% Loop over each file name to construct the model description
loaded_data = load('top_results/lap_logit7.mat');
% 1) Create a logical index for valid subjects
valid_subjects = ~isnan(loaded_data.cbm.math.logdetA) ...
    & ~isinf(loaded_data.cbm.math.logdetA) ...
    & (loaded_data.cbm.math.logdetA ~= 0) ...
    & imag(loaded_data.cbm.math.loglik) == 0 ...
    & imag(loaded_data.cbm.math.lme) == 0;
% Calculate the mean and SD of logdetA for valid subjects
q25 = quantile(loaded_data.cbm.math.loglik(valid_subjects),0.25);
q75 = quantile(loaded_data.cbm.math.loglik(valid_subjects),0.75);
% Add the condition for logdetA values within 3 SDs of the mean
valid_subjects = valid_subjects & ...
loaded_data.cbm.math.loglik >= q25 - 1.5*(q75-q25) & ...
loaded_data.cbm.math.loglik <= q75 + 1.5*(q75-q25);
valid_subj_all = valid_subj_all & valid_subjects;


% Aggregate only valid subjects
filtered_data = data(valid_subj_all);
num_subjects = length(filtered_data);
cbm_lap_aggregate(fname_subjs(valid_subj_all), ...
    'top_results/lap_ac15.mat');
fname_subjs = cell(num_subjects,1);
for subj = 1:length(data)
    % Construct the filename for the current subject's results
    fname_subjs{subj} = sprintf('top_results/logit/lap_logit7_%d.mat', subj);
end
cbm_lap_aggregate(fname_subjs(valid_subj_all), ...
    'top_results/lap_logit7.mat');


%% HBI

cbm_hbi(filtered_data, {@logit_wrapper_7, @wrapper_function_15}, ...
    {'top_results/lap_logit7.mat','top_results/lap_ac15.mat', }, ...
    'top_results/hbi_ac.mat');
cbm_hbi_null(filtered_data, 'top_results/hbi_ac.mat');
