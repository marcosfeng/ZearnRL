rng(37909890)
% © 1998-2023 RANDOM.ORG
% Timestamp: 2023-10-04 18:38:28 UTC

% Add path to the codes directory
addpath(fullfile('..','codes'));
addpath(fullfile('..','zearn_codes'));
addpath(fullfile('..','zearn_codes','hybrid_wrappers'));

% Load the common data for all datasets
fdata = load('../data/all_data.mat');
data  = fdata.data;

%% Estimate Models
% Temporarily add path, create handle, then remove path
wrapperPath = addpath(fullfile('..','zearn_codes','ql_wrappers'));
wrapper_function_7_ql = str2func('wrapper_function_7');
wrapper_function_1_ql = str2func('wrapper_function_1');
path(wrapperPath);
% Define function handles for Actor-Critic wrapper functions
wrapperPath = addpath(fullfile('..','zearn_codes','ac_wrappers'));
wrapper_function_2_ac = str2func('wrapper_function_2');
wrapper_function_51_ac = str2func('wrapper_function_51');
path(wrapperPath);

fname_template = {'comp_results/lap_logit7_%d.mat', ...
    'comp_results/lap_logit1_%d.mat', ...
    'comp_results/lap_logit2_%d.mat', ...
    'comp_results/lap_logit51_%d.mat', ...
    'comp_results/lap_ql7_%d.mat', ... % Q-learning
    'comp_results/lap_ql1_%d.mat', ...
    'comp_results/lap_ac2_%d.mat', ... % Actor-Critic
    'comp_results/lap_ac51_%d.mat', ...
    'comp_results/lap_hybrid7_%d.mat', ... % Hybrid
    'comp_results/lap_hybrid1_%d.mat', ... 
    'comp_results/lap_hybrid2_%d.mat', ... 
    'comp_results/lap_hybrid51_%d.mat', ... 
    'comp_results/lap_hybrid1_2_%d.mat', ... 
    'comp_results/lap_hybrid7_51_%d.mat'};
models = {@logit_wrapper_7, ...
    @logit_wrapper_1, ...
    @logit_wrapper_2, ...
    @logit_wrapper_51, ...
    wrapper_function_7_ql, ...
    wrapper_function_1_ql, ...
    wrapper_function_2_ac, ...
    wrapper_function_51_ac, ...
    @hybrid_wrapper_7, ...
    @hybrid_wrapper_1, ...
    @hybrid_wrapper_2, ...
    @hybrid_wrapper_51, ...
    @hybrid_wrapper_1_2, ...
    @hybrid_wrapper_7_51};
num_parameters = [8 * ones(1,2), 10, 12, ...
    4 * ones(1, 2), 7 * ones(1, 2), ...
    13 * ones(1, 2), 18, 20, 12 * ones(1, 2)];

% Define the prior variance
v = 6.25;
num_subjects = length(data);
priors = struct([]);
for i = 1:length(num_parameters)
    priors{i} = struct('mean', zeros(num_parameters(i), 1), ...
                       'variance', v);
end

% Create the PCONFIG struct
pconfig = struct();
pconfig.numinit = min(70*max(num_parameters), 1000);
pconfig.numinit_med = 1000;
pconfig.numinit_up = 10000;
pconfig.tolgrad = 1e-4;
pconfig.tolgrad_liberal = 0.01;
parfor i = 1:(length(num_parameters)*num_subjects)
    model_idx = floor((i-1)/num_subjects) + 1;
    subj_idx = mod(i-1, num_subjects) + 1;

    % Construct filename for saving output
    fname = sprintf(fname_template{model_idx}, subj_idx);
    if exist(fname,"file") == 2
        continue;
    end

    % Run the cbm_lap function for the current model and subject
    cbm_lap(data(subj_idx), models{model_idx}, ...
        priors{model_idx}, fname, pconfig);
end

% Pre-allocate structures to store aggregated results
fname_subjs = cell(num_subjects,length(models));
fname = {'comp_aggr_results/lap_logit7.mat', ...
    'comp_aggr_results/lap_logit1.mat', ...
    'comp_aggr_results/lap_logit2.mat', ...
    'comp_aggr_results/lap_logit51.mat', ...
    'comp_aggr_results/lap_ql7.mat', ... % Q-learning
    'comp_aggr_results/lap_ql1.mat', ...
    'comp_aggr_results/lap_ac2.mat', ... % Actor-Critic
    'comp_aggr_results/lap_ac51.mat', ...
    'comp_aggr_results/lap_hybrid7.mat', ... % Hybrid
    'comp_aggr_results/lap_hybrid1.mat', ... 
    'comp_aggr_results/lap_hybrid2.mat', ... 
    'comp_aggr_results/lap_hybrid51.mat', ... 
    'comp_aggr_results/lap_hybrid1_2.mat', ... 
    'comp_aggr_results/lap_hybrid7_51.mat'};
for m = 1:length(models)
    % Aggregate results for each subject
    for subj = 1:num_subjects
        % Construct the filename for the current subject's results
        fname_subjs{subj,m} = sprintf(fname_template{m}, subj);
    end
    cbm_lap_aggregate(fname_subjs(:,m),fname{m});
end

valid_subj_all = ones(1,num_subjects);
% Loop over each file name to construct the model description
for i = 1:length(fname)
    loaded_data = load(fname{i});

    % 1) Create a logical index for valid subjects
    valid_subjects = ~isnan(loaded_data.cbm.math.logdetA) ...
        & ~isinf(loaded_data.cbm.math.logdetA) ...
        & (loaded_data.cbm.math.logdetA ~= 0) ...
        & imag(loaded_data.cbm.math.loglik) == 0 ...
        & imag(loaded_data.cbm.math.lme) == 0;
    % Calculate the mean and SD of logdetA for valid subjects
    mean_loglik = mean(loaded_data.cbm.math.loglik(valid_subjects));
    std_loglik = std(loaded_data.cbm.math.loglik(valid_subjects));
    % Add the condition for logdetA values within 3 SDs of the mean
    valid_subjects = valid_subjects & ...
        (abs(loaded_data.cbm.math.loglik - mean_loglik) <= ...
        3 * std_loglik);
    valid_subj_all = valid_subj_all & valid_subjects;
end

% Aggregate only valid subjects
filtered_data = data(valid_subj_all);
num_subjects = length(filtered_data);
fname_subjs = fname_subjs(valid_subj_all,:);
for m = 1:length(models)
    % Aggregate results for each subject
    cbm_lap_aggregate(fname_subjs(:,m),fname{m});
end

%% Model comparison (with top AC)

fname_hbi = {'hbi_compare_7.mat', ...
    'hbi_compare_1.mat', ...
    'hbi_compare_2.mat', ...
    'hbi_compare_51.mat'};
idx = [1,5,9,14;
    2,6,10,13;
    3,7,11,13;
    4,8,12,14];

pconfig = struct();
pconfig.verbose = 1;
pconfig.parallel = 1;
cbm_hbi(filtered_data, models(idx(i,4)), fname(idx(i,4)), 'hbi_compare_TEST.mat', pconfig)
parfor i = 1:size(idx,1)
    cbm_hbi(filtered_data, models(idx(i,:)), fname(idx(i,:)), fname_hbi{i});
end
parfor i = 1:size(idx,1)
    cbm_hbi_null(filtered_data, fname_hbi{i});
end

% [p_sf,stats_sf] = cbm_hbi_ttest(fname_hbi,3,0,1);

% param_names = {'\alpha','\gamma', '\tau', ...
%     'c_1', 'c_2', 'c_3'};
% transform = {'sigmoid','sigmoid','exp',...
%     'exp','exp','exp'};
% model_names = {'Logit', 'Kernel', 'Actor-Critic'};
% % Load the HBI results and display them
% cbm_hbi_plot(fname_hbi,model_names, param_names,transform,2);
