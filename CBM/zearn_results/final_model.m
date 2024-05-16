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
    @logit_wrapper_3, ...
    @logit_wrapper_7, ...
    @logit_wrapper_51s, ...
    @logit_wrapper_13s, ...
    @ql_wrapper_3, ...
    @ql_wrapper_7, ...
    @ac_wrapper_51, ...
    @ac_wrapper_13, ...
    @hybrid_wrapper_3, ...
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

%% Estimate all relevant models

% Indeces:
idx = [1,2,3, ... % Logit
    5, 6, ...  % QL
    7];     % AC
fname_template = {
    'top_results/logit/lap_logit3_%d.mat', ... % Logit
    'top_results/logit/lap_logit7_%d.mat', ...
    'top_results/logit/lap_logit51s_%d.mat', ...
    'top_results/ql/lap_ql3_%d.mat', ...  % Q-learning
    'top_results/ql/lap_ql7_%d.mat', ...  
    'top_results/ac/lap_ac51_%d.mat'}; % Actor-Critic

% Define the prior variance
v = 6.25;
num_subjects = length(data);
priors = struct([]);
for i = 1:length(num_parameters(idx))
    priors{i} = struct('mean', zeros(num_parameters(idx(i)), 1), ...
                       'variance', v);
end

% PCONFIG structure with refined setup (with multiplier)
mult = 8;
pconfig = struct();
pconfig.numinit = min(7 * max(num_parameters), 100) * mult;
pconfig.numinit_med = 70 * mult;
pconfig.numinit_up = 100 * mult;
pconfig.tolgrad = .001001 / mult;
pconfig.tolgrad_liberal = .1 / mult;
pconfig.prior_for_bads = 0;

success = nan(num_subjects*length(priors),1);
parfor i = 1:(length(num_parameters(idx))*num_subjects)
    model_idx = floor((i-1)/num_subjects) + 1;
    subj_idx = mod(i-1, num_subjects) + 1;

    % Construct filename for saving output
    fname = sprintf(fname_template{model_idx}, subj_idx);
    % % if you need to re-run models
    % if exist(fname,"file") == 2
    %     success(i) = 1;
    %     continue
    % end

    % Run the cbm_lap function for the current model and subject
    [~, success(i)] = ...
        cbm_lap(data(subj_idx), models{idx(model_idx)}, ...
        priors{model_idx}, fname, pconfig);
end
success = reshape(success,[num_subjects,length(priors)]);
success = logical(success);
save("top_results/success.mat","success");

%% HBI

load("top_results/success.mat");
% Check data is the same size:
% sum(success)

fname_hbi = {'top_results/hbi_compare_3.mat', ...
    'top_results/hbi_compare_7.mat', ...
    'top_results/hbi_compare_51.mat', ...
    'top_results/hbi_compare_7_51.mat'};
hbi_idx = [1,4;2,5;3,6;5,6];

% Pre-allocate structures to store aggregated results
fname_subjs = cell(num_subjects,length(fname_template));
for subj = 1:num_subjects
    % Construct the filename for the current subject's results
    fname_subjs(subj,:) = [compose(fname_template, subj)];
end

pconfig = struct();
pconfig.maxiter = 200;
parfor i = 1:size(hbi_idx,1)
    success_filter = all(success(:,hbi_idx(i,:)),2);
    % Aggregate results for each subject
    for m = 1:size(hbi_idx,2)
        cbm_lap_aggregate( ...
            fname_subjs(success_filter,hbi_idx(i,m)), ...
            sprintf( ...
            replace(fname_template{hbi_idx(i,m)},"/lap_", "_aggr_"), ...
            i));
    end
    % % if you need to re-run models
    % if exist(fname_hbi{i},"file") == 2
    %     continue
    % end
    cbm_hbi(data(success_filter), ...
        models(idx(hbi_idx(i,:))), ...
        compose( ...
            replace(fname_template(hbi_idx(i,:)),"/lap_", "_aggr_"), ...
        i), ...
        fname_hbi{i}, pconfig);
    cbm_hbi_null(data(success_filter), fname_hbi{i});
end

%% Posteriors

prob = cell(length(data),numel(hbi_idx));
auc = nan(length(data),numel(hbi_idx));
auc_conf = nan(length(data),numel(hbi_idx));
roc = cell(length(data),numel(hbi_idx));
loglik = nan(length(data),numel(hbi_idx));

for posterior_idx = 1:size(hbi_idx,1)
    hbi_model = load(fname_hbi{posterior_idx});
    success_filter = all(success(:,hbi_idx(posterior_idx,:)),2);
    for model_idx = 1:size(hbi_idx,2)
        wrapper = str2func( ...
            sprintf('wrapper_post_%d', ...
            idx(hbi_idx(posterior_idx,model_idx))));
    
        for j = 1:length(data)
            if ~success_filter(j), continue, end
            [loglik(j,((posterior_idx-1)*size(hbi_idx,2)+model_idx)), ...
                prob{j,((posterior_idx-1)*size(hbi_idx,2)+model_idx)}, ...
                choice] = wrapper( ...
                hbi_model.cbm.output.parameters{model_idx}( ...
                j - sum(~success_filter(1:j)),:), ...
                data{j});
            roc{j,((posterior_idx-1)*size(hbi_idx,2)+model_idx)} = ...
                rocmetrics(choice, ...
                prob{j,((posterior_idx-1)*size(hbi_idx,2)+model_idx)}, [0,1], ...
                NumBootstraps=100,BootstrapOptions=statset(UseParallel=true));
            auc(j,((posterior_idx-1)*size(hbi_idx,2)+model_idx)) = ...
                roc{j,((posterior_idx-1)*size(hbi_idx,2)+model_idx)}.AUC(1,1);
            auc_conf(j,((posterior_idx-1)*size(hbi_idx,2)+model_idx)) = ...
                roc{j,((posterior_idx-1)*size(hbi_idx,2)+model_idx)}.AUC(3,1) - ...
                roc{j,((posterior_idx-1)*size(hbi_idx,2)+model_idx)}.AUC(2,1);
        end
    end
end
auc_matrix = reshape(mean(auc,"omitmissing"),size(hbi_idx'))';

for posterior_idx = 1:size(hbi_idx,1)
    load(fname_hbi{posterior_idx});
    cbm.output.auc = ...
        auc(:,((posterior_idx-1)*size(hbi_idx,2))+(1:size(hbi_idx,2)));
    cbm.output.auc_conf = ...
        auc_conf(:,((posterior_idx-1)*size(hbi_idx,2))+(1:size(hbi_idx,2)));
    cbm.output.loglik = ...
        loglik(:,((posterior_idx-1)*size(hbi_idx,2))+(1:size(hbi_idx,2)));
    save(fname_hbi{posterior_idx},"cbm");
end
