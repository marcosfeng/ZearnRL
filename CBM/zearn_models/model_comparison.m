rng(37909890)
% © 1998-2023 RANDOM.ORG
% Timestamp: 2023-10-04 18:38:28 UTC

% Add path to the codes directory
addpath(fullfile('..','codes'));

% Add path to the wrappers directory
addpath(fullfile('..','zearn_models','wrappers'));

% Load the common data for all datasets
fdata = load('../data/all_data.mat');
data  = fdata.data;

%% Estimate Models

% all_alerts = [];
% for i = 1:length(data)
%     all_alerts = [all_alerts; data{i,1}.alerts];
% end
% global_median = median(all_alerts);

fname_template = {'subj_results/lap_logit_%d.mat', ...
    'subj_results/lap_kernel_%d.mat', ...
    'subj_results/lap_ac_%d.mat'};
models = {@logit_model, ...
    @q_learning_model, ...
    @wrapper_function_39};
num_parameters = [3*4, 6, 9];

% Define the prior variance
v = 6.25;
num_subjects = length(data);
priors = {struct('mean', zeros(num_parameters(1), 1), 'variance', v), ...
    struct('mean', zeros(num_parameters(2), 1), 'variance', v), ...
    struct('mean', zeros(num_parameters(3), 1), 'variance', v)};
% Create the PCONFIG struct
pconfig = struct();
pconfig.numinit = min(70*max(num_parameters), 1000);
pconfig.numinit_med = 1000;
pconfig.numinit_up = 10000;
pconfig.tolgrad = 1e-4;
pconfig.tolgrad_liberal = 0.01;
parfor i = 1:(3*num_subjects)
    model_idx = floor((i-1)/num_subjects) + 1;
    subj_idx = mod(i-1, num_subjects) + 1;
    
    % Construct filename for saving output
    fname = sprintf(fname_template{model_idx}, subj_idx);
    
    % Run the cbm_lap function for the current model and subject
    cbm_lap(data(subj_idx), models{model_idx}, ...
        priors{model_idx}, fname, pconfig);
end

% Pre-allocate structures to store aggregated results
fname_subjs = cell(num_subjects,length(models));
fname = {'subj_results/lap_logit_all.mat', ...
    'subj_results/lap_kernel_all.mat', ...
    'subj_results/lap_ac_all.mat'};
for m = 1:length(models)
    % Aggregate results for each subject
    for subj = 1:num_subjects
        % Construct the filename for the current subject's results
        fname_subjs{subj,m} = sprintf(fname_template{m}, subj);
    end
    cbm_lap_aggregate(fname_subjs(:,m),fname{m});
end

%% Model comparison (with top AC)

fname_hbi = 'hbi_model_compare.mat';
cbm_hbi(data, models, fname, fname_hbi);
% save('../data/filtered_data.mat', 'filtered_data');
cbm_hbi_null(data, fname_hbi);

[p,stats] = cbm_hbi_ttest(fname_hbi,3,0,1);

fname_hbi_loaded = load(fname_hbi);
param_names = {'\alpha','\gamma', '\tau', ...
    'c_1', 'c_2', 'c_3'};
% note the latex format
% transformation functions associated with each parameter
transform = {'sigmoid','sigmoid','exp',...
    'exp','exp','exp'};
model_names = {'Logit', 'Kernel', 'Actor-Critic'};

% Load the HBI results and display them
cbm_hbi_plot(fname_hbi,model_names, param_names, transform);
