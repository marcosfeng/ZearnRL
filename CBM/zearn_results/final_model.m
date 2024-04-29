rng(37909890)
% © 1998-2023 RANDOM.ORG
% Timestamp: 2023-10-04 18:38:28 UTC

% Add path to the codes directory
addpath(fullfile('..','codes'));
addpath(fullfile('..','zearn_codes'));
addpath(fullfile('..','zearn_codes','hybrid_wrappers'));

% Load the common data for all datasets
fdata = load('../data/full_data.mat');
data  = fdata.data;

%% Models

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

%% Estimate all relevant models

% Indeces:
idx = [1, ... % Logit
    5,6, ...  % QL
    7,8];     % AC
fname_template = {
    'top_results/logit/lap_logit7_%d.mat', ... % Logit
    'top_results/ql/lap_ql7_%d.mat', ...  % Q-learning
    'top_results/ql/lap_ql3_%d.mat', ...
    'top_results/ac/lap_ac44_%d.mat', ... % Actor-Critic
    'top_results/ac/lap_ac3_%d.mat'};

% Define the prior variance
v = 6.25;
priors = struct([]);
for i = 1:length(num_parameters(idx))
    priors{i} = struct('mean', zeros(num_parameters(idx(i)), 1), ...
                       'variance', v);
end

num_subjects = length(data);
% Create the PCONFIG struct
pconfig = struct();
pconfig.numinit = min(70*max(num_parameters(idx)), 1000);
pconfig.numinit_med = 1000;
pconfig.numinit_up = 10000;
pconfig.tolgrad = 1e-4;
pconfig.tolgrad_liberal = 0.01;
parfor i = 1:(length(num_parameters(idx))*num_subjects)
    model_idx = floor((i-1)/num_subjects) + 1;
    subj_idx = mod(i-1, num_subjects) + 1;

    % Construct filename for saving output
    fname = sprintf(fname_template{model_idx}, subj_idx);
    % if you need to re-run models
    if exist(fname,"file") == 2
        continue;
    end

    % Run the cbm_lap function for the current model and subject
    cbm_lap(data(subj_idx), models{idx(model_idx)}, ...
        priors{model_idx}, fname, pconfig);
end

% Pre-allocate structures to store aggregated results
fname_subjs = cell(num_subjects,length(models(idx)));
fname = {
    'top_results/lap_logit7.mat', ... % Logit
    'top_results/lap_ql7.mat', ...  % Q-learning
    'top_results/lap_ql3.mat', ...
    'top_results/lap_ac44.mat', ... % Actor-Critic
    'top_results/lap_ac3.mat'};
% Aggregate results
for m = 1:length(models(idx))
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
num_subjects = length(filtered_data);
fname_subjs = fname_subjs(valid_subj_all,:);
for m = 1:length(models(idx))
    % Aggregate results for each subject
    cbm_lap_aggregate(fname_subjs(:,m),fname{m});
end

%% Compare between QL and logit models

fname_hbi = {'top_results/hbi_compare_ql.mat', ...
    'top_results/hbi_compare_logit_ql7.mat',
    'top_results/hbi_compare_logit_ql3.mat'};
% Indeces for comparisons
idx = [5,6; % Compare QL models
    1,5;
    1,6];   % Compare AC models
idx_fname = [2,3;1,2;1,3];
parfor i = 1:size(idx,1)
    cbm_hbi(filtered_data, models(idx(i,:)), ...
        fname(idx_fname(i,:)), fname_hbi{i});
    cbm_hbi_null(filtered_data, fname_hbi{i});
end


%% Compare between top QL and top AC

fname_hbi = {'top_results/hbi_compare_ql_ac.mat', ...
    'top_results/hbi_compare_ac.mat'};
% Indeces for comparisons
idx = [,; % Compare QL models
    7,8];   % Compare AC models
idx_fname = [,;1,];
parfor i = 1:size(idx,1)
    cbm_hbi(filtered_data, models(idx(i,:)), ...
        fname(idx_fname(i,:)), fname_hbi{i});
    cbm_hbi_null(filtered_data, fname_hbi{i});
end

%% Plotting

fname_hbi = 'top_results/hbi_ac.mat';

cbm = load(fname_hbi).cbm;
cbm.output

model_names = {'Logit', 'Actor-Critic'};
param_names = {'\alpha_w','\alpha_\theta', '\gamma', ...
    '\tau', '\theta_{t=1}', 'w_{t=1}', 'Cost'};
transform = {'sigmoid','sigmoid','sigmoid',...
    'exp', 'none', 'none', 'exp'};

cbm_hbi_plot(fname_hbi, model_names, param_names, transform)

%% Posterior predictive check

fname_hbi = 'top_results/hbi_ac.mat';
cbm = load(fname_hbi).cbm;

loglik = zeros(length(filtered_data), 1);
prob = cell(length(filtered_data), 1);
theta = cell(length(filtered_data), 1);
w = cell(length(filtered_data), 1);

for i = 1:length(filtered_data)
    subj = filtered_data{i};
    subj.outcome = subj.NNDSVD_student2;
    % Action variable: NNDSVD_teacher2
    subj.action = subj.NNDSVD_teacher2;
    % State variables: NNDSVD_student1 
    subj.state = [subj.NNDSVD_student1 ];
    [loglik(i),prob{i},theta{i},w{i}] = ...
        actor_critic_posterior(cbm.output.parameters{2, 1}(i,:), subj);
end


st_test = [ones(length(subj.action), 1), subj.state];
st_test(7, :) * theta(:,7) * exp(cbm.output.parameters{2, 1}(1,4))