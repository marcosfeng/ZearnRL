rng(37909890)
% © 1998-2023 RANDOM.ORG
% Timestamp: 2023-10-04 18:38:28 UTC

% Add path to the codes directory
addpath(fullfile('..','codes'));
addpath(fullfile('..','zearn_codes'));
addpath(fullfile('..','zearn_codes','hybrid_wrappers'));

% Load the common data for all datasets
fdata = load('../data/sample_data.mat');
data  = fdata.data;

%% Estimate Models

fname_template = {
    'comp_results/lap_logit7_%d.mat', ... % Logit
    'comp_results/lap_logit3_%d.mat', ...
    'comp_results/lap_logit44s_%d.mat', ...
    'comp_results/lap_logit3s_%d.mat', ...
    'comp_results/lap_ql7_%d.mat', ... % Q-learning
    'comp_results/lap_ql3_%d.mat', ...
    'comp_results/lap_ac44_%d.mat', ... % Actor-Critic
    'comp_results/lap_ac3_%d.mat', ...
    'comp_results/lap_hybrid7_%d.mat', ... % Logit Hybrid
    'comp_results/lap_hybrid3_%d.mat', ... 
    'comp_results/lap_hybrid44s_%d.mat', ... 
    'comp_results/lap_hybrid3s_%d.mat', ... 
    'comp_results/lap_hybrid7_44_%d.mat', ... % RL Hybrid
    'comp_results/lap_hybrid7_3_%d.mat'};
models = {
    @logit_wrapper_7, ...
    @logit_wrapper_3, ...
    @logit_wrapper_44s, ...
    @logit_wrapper_3s, ...
    @ql_wrapper_7, ...
    @ql_wrapper_3, ...
    @ac_wrapper_44, ...
    @ac_wrapper_3, ...
    @hybrid_wrapper_7, ...
    @hybrid_wrapper_3, ...
    @hybrid_wrapper_44s, ...
    @hybrid_wrapper_3s, ...
    @hybrid_wrapper_7_44, ...
    @hybrid_wrapper_7_3};
num_parameters = [
    8  * ones(1,2), 10 * ones(1,2), ...
    4  * ones(1,2),  7 * ones(1,2), ...
    13 * ones(1,2), 18 * ones(1,2), ...
    12 * ones(1,2)];

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
    % % if you need to re-run models
    % if exist(fname,"file") == 2
    %     continue;
    % end

    % Run the cbm_lap function for the current model and subject
    cbm_lap(data(subj_idx), models{model_idx}, ...
        priors{model_idx}, fname, pconfig);
end

% Pre-allocate structures to store aggregated results
fname_subjs = cell(num_subjects,length(models));
fname = {'comp_aggr_results/lap_logit7.mat', ...
    'comp_aggr_results/lap_logit3.mat', ...
    'comp_aggr_results/lap_logit44s.mat', ...
    'comp_aggr_results/lap_logit3s.mat', ...
    'comp_aggr_results/lap_ql7.mat', ... % Q-learning
    'comp_aggr_results/lap_ql3.mat', ...
    'comp_aggr_results/lap_ac44.mat', ... % Actor-Critic
    'comp_aggr_results/lap_ac3.mat', ...
    'comp_aggr_results/lap_hybrid7.mat', ... % Hybrid
    'comp_aggr_results/lap_hybrid3.mat', ... 
    'comp_aggr_results/lap_hybrid44s.mat', ... 
    'comp_aggr_results/lap_hybrid3s.mat', ... 
    'comp_aggr_results/lap_hybrid7_44.mat', ... 
    'comp_aggr_results/lap_hybrid7_3.mat'};

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
    q25 = quantile(loaded_data.cbm.math.loglik(valid_subjects),0.25);
    q75 = quantile(loaded_data.cbm.math.loglik(valid_subjects),0.75);
    % Add the condition for logdetA values within 3 SDs of the mean
    valid_subjects = valid_subjects & ...
        loaded_data.cbm.math.loglik >= q25 - 1.5*(q75-q25) & ...
        loaded_data.cbm.math.loglik <= q75 + 1.5*(q75-q25);
    valid_subj_all = valid_subj_all & valid_subjects;
end

% Aggregate only valid subjects
filtered_data = data(valid_subj_all);
if num_subjects == length(filtered_data), equilibrium = true; end
num_subjects = length(filtered_data);
fname_subjs = fname_subjs(valid_subj_all,:);
for m = 1:length(models)
    % Aggregate results for each subject
    cbm_lap_aggregate(fname_subjs(:,m),fname{m});
end

%% Model comparison (with top AC)

fname_hbi = {'comp_aggr_results/hbi_compare_7.mat', ...
    'comp_aggr_results/hbi_compare_3.mat', ...
    'comp_aggr_results/hbi_compare_44s.mat', ...
    'comp_aggr_results/hbi_compare_3s.mat'};
idx = [1,5,9,14;
    2,6,10,13;
    3,7,11,14;
    4,8,12,13];

parfor i = 1:size(idx,1)
    cbm_hbi(filtered_data, models(idx(i,:)), fname(idx(i,:)), fname_hbi{i});
    cbm_hbi_null(filtered_data, fname_hbi{i});
end

top_idx = nan(1,length(idx));
% Load the HBI results and store them
for i = 1:size(idx,1)
    fname_hbi_loaded = load(fname_hbi{i});
    hbi_results = fname_hbi_loaded.cbm;
    hbi_results.output
    % Top model
    [~, j] = ...
        max(hbi_results.output.protected_exceedance_prob);
    top_idx(i) = idx(i,j);
end

% Which of the top models fit best
cbm_hbi(filtered_data, models(top_idx), fname(top_idx), ...
    'comp_aggr_results/top_model_comp.mat');
cbm_hbi_null(filtered_data, 'comp_aggr_results/top_model_comp.mat');



fname_hbi_loaded = load(fname_hbi{4});
means = fname_hbi_loaded.cbm.output.group_mean{2};
means(1:3) = 1./(1+exp(-means(1:3)));
means([4,7]) = exp(means([4,7]));
display(means)

fname_hbi_loaded = load(fname_hbi{3});
means = fname_hbi_loaded.cbm.output.group_mean{2};
means(1:3) = 1./(1+exp(-means(1:3)));
means([4,7]) = exp(means([4,7]));
display(means)

transform = {'sigmoid','sigmoid','sigmoid', ...
    'exp', 'none', 'none', 'exp'};

% [p_sf,stats_sf] = cbm_hbi_ttest(fname_hbi{1},3,0,1);
model_names = {'Logit', 'Q-Learning', ...
    'Hybrid Logit', 'Hybrid AC'};
param_names = {'Intercept','R_{t-1}','R_{t-2}', ...
    'A_{t-1}','A_{t-2}', ...
    'R_{t-1}xA_{t-1}', 'R_{t-1}xA_{t-2}', 'R_{t-2}xA_{t-2}'}; 
transform = {'none','none','none', ...
    'none', 'none', ...
    'none', 'none', 'none'};
cbm_hbi_plot(fname_hbi{1}, model_names,param_names,transform);
