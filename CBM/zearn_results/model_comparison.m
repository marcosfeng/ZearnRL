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
    'comp_results/lap_logit1_%d.mat', ... % Logit
    'comp_results/lap_logit7_%d.mat', ...
    'comp_results/lap_logit51s_%d.mat', ...
    'comp_results/lap_logit13s_%d.mat', ...
    'comp_results/lap_ql1_%d.mat', ... % Q-learning
    'comp_results/lap_ql7_%d.mat', ...
    'comp_results/lap_ac51_%d.mat', ... % Actor-Critic
    'comp_results/lap_ac13_%d.mat', ...
    'comp_results/lap_hybrid1_%d.mat', ... % Logit Hybrid
    'comp_results/lap_hybrid7_%d.mat', ... 
    'comp_results/lap_hybrid51_%d.mat', ... 
    'comp_results/lap_hybrid13_%d.mat', ... 
    'comp_results/lap_hybrid7_51_%d.mat', ... % RL Hybrid
    'comp_results/lap_hybrid1_13_%d.mat'};
models = {
    @logit_wrapper_1, ...
    @logit_wrapper_7, ...
    @logit_wrapper_51s, ...
    @logit_wrapper_13s, ...
    @ql_wrapper_1, ...
    @ql_wrapper_7, ...
    @ac_wrapper_51, ...
    @ac_wrapper_13, ...
    @hybrid_wrapper_1, ...
    @hybrid_wrapper_7, ...
    @hybrid_wrapper_51s, ...
    @hybrid_wrapper_13s, ...
    @hybrid_wrapper_7_51, ...
    @hybrid_wrapper_1_13};
num_parameters = [
    8  * ones(1,2), 12 * ones(1,1), 14 * ones(1,1), ...
    5  * ones(1,2), 11 * ones(1,1), 13 * ones(1,1), ...
    14 * ones(1,2), 24 * ones(1,1), 28 * ones(1,1), ...
    17 * ones(1,1), 19 * ones(1,1)];

% Define the prior variance
v = 6.25;
num_subjects = length(data);
priors = struct([]);
for i = 1:length(num_parameters)
    priors{i} = struct('mean', zeros(num_parameters(i), 1), ...
                       'variance', v);
end

% PCONFIG structure with refined setup (with multiplier)
mult = 4;
pconfig = struct();
pconfig.numinit = min(7 * max(num_parameters), 100) * mult;
pconfig.numinit_med = 70 * mult;
pconfig.numinit_up = 100 * mult;
pconfig.tolgrad = .001001 / mult;
pconfig.tolgrad_liberal = .1 / mult;
pconfig.prior_for_bads = 0;

success = nan(num_subjects*length(num_parameters),1);
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
    [~, success(i)] = ...
        cbm_lap(data(subj_idx), models{model_idx}, ...
        priors{model_idx}, fname, pconfig);
end
success = reshape(success,[num_subjects,length(num_parameters)]);
success = logical(success);
save("comp_results/success.mat","success");

%% HBI

load("comp_results/success.mat");

fname_hbi = {'comp_results/hbi/hbi_compare_1.mat', ...
    'comp_results/hbi/hbi_compare_7.mat', ...
    'comp_results/hbi/hbi_compare_51s.mat', ...
    'comp_results/hbi/hbi_compare_13s.mat', ...
    'comp_results/hbi/hbi_compare_7_51s.mat', ...
    'comp_results/hbi/hbi_compare_1_13s.mat'};
idx = [1,5,9;
    2,6,10;
    3,7,11;
    4,8,12;
    6,7,13;
    5,8,14];

% Pre-allocate structures to store aggregated results
fname_subjs = cell(num_subjects,length(fname_template));
for subj = 1:num_subjects
    % Construct the filename for the current subject's results
    fname_subjs(subj,:) = [compose(fname_template, subj)];
end
parfor i = 1:size(idx,1)
    success_filter = all(success(:,idx(i,:)),2);
    % Aggregate results for each subject
    for m = 1:size(idx,2)
        cbm_lap_aggregate( ...
            fname_subjs(success_filter,idx(i,m)), ...
            sprintf( ...
            replace(fname_template{idx(i,m)},"lap_", "lap_aggr_"), ...
            i)); % Mark aggregate with 0
    end
    cbm_hbi(data(success_filter), ...
        models(idx(i,:)), ...
        compose( ...
            replace(fname_template(idx(i,:)),"lap_", "lap_aggr_"), ...
        i), ...
        fname_hbi{i});
    cbm_hbi_null(data(success_filter), fname_hbi{i});
end


% Top models
top_idx = nan(1,length(idx));
% Load the HBI results and store them
for i = 1:size(idx,1)
    fname_hbi_loaded = load(fname_hbi{i});
    hbi_results = fname_hbi_loaded.cbm;
    % Top model
    [~, j] = ...
        max(hbi_results.output.protected_exceedance_prob);
    top_idx(i) = idx(i,j);
end

%% Posteriors

prob = struct([]);
auc = nan(length(data(success)),numel(idx));
roc = struct([]);
loglik = nan(length(data(success)),numel(idx));
for hbi_idx = 1:size(idx,1)
    hbi_model = load(fname_hbi{hbi_idx});

    for model_idx = 1:size(idx,2)
        wrapper = str2func( ...
            sprintf('wrapper_post_%d', idx(hbi_idx,model_idx)));

        for j = 1:length(data)
            if ~success(j), continue, end
            [loglik(j,((hbi_idx-1)*size(idx,2)+model_idx)), ...
                prob{j,((hbi_idx-1)*size(idx,2)+model_idx)}, ...
                choice] = wrapper( ...
                hbi_model.cbm.output.parameters{model_idx}( ...
                j - sum(~success(1:j)),:), ...
                data{j});
            roc{j,((hbi_idx-1)*size(idx,2)+model_idx)} = ...
                rocmetrics(choice, ...
                prob{j,((hbi_idx-1)*size(idx,2)+model_idx)}, ...
                [0,1]);
            auc(j,((hbi_idx-1)*size(idx,2)+model_idx)) = ...
                roc{j,((hbi_idx-1)*size(idx,2)+model_idx)}.AUC(1);
        end
    end
end

mean(auc,"omitmissing")

