rng(37909890)
% © 1998-2023 RANDOM.ORG
% Timestamp: 2023-10-04 18:38:28 UTC

% Add path to the codes directory
addpath(fullfile('..','codes'));

% Add path to the wrappers directory
addpath(fullfile('..','zearn_models','wrappers'));

% Define the prior variance
v = 6.25;
% Determine the number of parameters in your model.
num_parameters = 3*5;
% Create the prior structure for your new model
prior_logit = struct('mean', zeros(num_parameters, 1), 'variance', v);

% Load the common data for all datasets
fdata = load('../data/all_data.mat');
data  = fdata.data;

% Specify the file-address for saving the output
fname = 'subj_results/lap_logit.mat';
cbm_lap(data, @logit_model, prior_logit, fname);

%% Q-learning Kernel

all_alerts = [];
for i = 1:length(data)
    all_alerts = [all_alerts; data{i,1}.alerts];
end
global_median = median(all_alerts);

% Determine the number of parameters in your model.
num_parameters = 6;
% Create the prior structure for your new model
prior_kernel = struct('mean', zeros(num_parameters, 1), 'variance', v);

% Specify the file-address for saving the output
fname = 'subj_results/lap_kernel.mat';
% Run the cbm_lap function for your new model
cbm_lap(data, @q_learning_model, prior_kernel, fname);

%% Model comparison (with top AC)

filtered_name = {'subj_results/filtered_logit.mat', ...
    'subj_results/filtered_kernel.mat', ...
    'ac_refine/filtered_ac.mat'};
fname = {'subj_results/lap_logit.mat', ...
    'subj_results/lap_kernel.mat', ...
    'ac_refine/refine_ac_24.mat'};
models = {@logit_model, ...
    @q_learning_model, ...
    @wrapper_function_24};

% Initialize a parallel pool if it doesn't already exist
if isempty(gcp('nocreate'))
    parpool;
end
num_parameters = {3*5, 6, 9};
parfor i = 1:3
    % Create the PCONFIG struct
    pconfig = struct();
    pconfig.numinit = min(140*num_parameters{i}, 2000);
    pconfig.numinit_med = 2000;
    pconfig.numinit_up = 20000;
    pconfig.tolgrad = 5e-5;
    pconfig.tolgrad_liberal = 0.005;
    % Create the prior structure for your new model
    prior = struct('mean', zeros(num_parameters{i}, 1), 'variance', v);
    % Run the cbm_lap function for your new model
    cbm_lap(data, models{i}, prior, filtered_name{i}, pconfig);
end


valid_subj_all = ones(1,210);
for i = 1:3
    loaded_data = load(fname{i});
    % 1) Create a logical index for valid subjects
    valid_subjects = ~isnan(loaded_data.cbm.math.logdetA) ...
        & ~isinf(loaded_data.cbm.math.logdetA) ...
        & (loaded_data.cbm.math.logdetA ~= 0);
    % Calculate the mean and standard deviation of logdetA for valid subjects
    mean_logdetA = mean(loaded_data.cbm.math.logdetA(valid_subjects));
    std_logdetA = std(loaded_data.cbm.math.logdetA(valid_subjects));
    % Add the condition for logdetA values within 3 standard deviations of the mean
    valid_subjects = valid_subjects & ...
        (abs(loaded_data.cbm.math.logdetA - mean_logdetA) <= 3 * std_logdetA);
    valid_subj_all = valid_subj_all & valid_subjects;
end
filtered_data = data(valid_subj_all);

for i = 1:3
    loaded_data = load(fname{i});
    % 2) Filtering
    loaded_data.cbm.profile.optim.flag = ...
        loaded_data.cbm.profile.optim.flag(valid_subj_all);
    loaded_data.cbm.profile.optim.gradient = ...
        loaded_data.cbm.profile.optim.gradient(:, valid_subj_all);
    loaded_data.cbm.math.A = ...
        loaded_data.cbm.math.A(valid_subj_all);
    loaded_data.cbm.math.Ainvdiag = ...
        loaded_data.cbm.math.Ainvdiag(valid_subj_all);
    loaded_data.cbm.math.lme = ...
        loaded_data.cbm.math.lme(valid_subj_all);
    loaded_data.cbm.math.logdetA = ...
        loaded_data.cbm.math.logdetA(valid_subj_all);
    loaded_data.cbm.math.loglik = ...
        loaded_data.cbm.math.loglik(valid_subj_all);
    loaded_data.cbm.math.theta = ...
        loaded_data.cbm.math.theta(valid_subj_all);
    loaded_data.cbm.output.log_evidence = ...
        loaded_data.cbm.output.log_evidence(valid_subj_all);
    loaded_data.cbm.output.parameters = ...
        loaded_data.cbm.output.parameters(valid_subj_all, :);

    % Save the modified loaded_data back to the same file
    save(filtered_name{i}, '-struct', 'loaded_data');
end
fname_hbi = 'hbi_model_compare.mat';
cbm_hbi(filtered_data, models, filtered_name, fname_hbi);
% save('../data/filtered_data.mat', 'filtered_data');

load(fname_hbi);
param_names = {'\alpha','\gamma', '\tau', ...
    'c_1', 'c_2', 'c_3'};
% note the latex format
% transformation functions associated with each parameter
transform = {'sigmoid','sigmoid','exp',...
    'exp','exp','exp'};
model_names = {'Logit', 'Kernel', 'Actor-Critic'};

% Load the HBI results and display them
cbm_hbi_plot(fname_hbi,model_names, param_names, transform);
