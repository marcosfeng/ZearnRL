rng(37909890)
% © 1998-2023 RANDOM.ORG
% Timestamp: 2023-10-04 18:38:28 UTC

% Add path to the codes directory
addpath(fullfile('..','codes'));
addpath(fullfile('..','zearn_codes'));
addpath(fullfile('..','zearn_codes','posterior_wrappers'));

% Load the common data for all datasets
fdata = load('../data/full_data.mat');
data  = fdata.data;

%% Models

fname_template = {
    'aggr_results/lap_aggr_baseline_model_2.mat', ... % Baseline
    'aggr_results/lap_aggr_logit_model_2.mat', ... % Logit
    'aggr_results/lap_aggr_q_model_2.mat'}; % Q-learning
models = {
    @posterior_baseline_model_2, ...
    @posterior_logit_model_2, ...
    @posterior_q_model_2};
num_parameters = [
    1, 5, 5];

%% HBI

fname_hbi = {'comp_results/hbi/hbi_compare.mat'};
idx = [1,3];

% Pre-allocate structures to store aggregated results
fname_subjs = cell(size(idx,1));
pconfig = struct();
pconfig.maxiter = 200;
parfor i = 1:size(idx,1)
    cbm_hbi(data, ...
        models(idx(i,:)), ...
        fname_template(idx(i,:)), ...
        fname_hbi{i}, pconfig);
    cbm_hbi_null(data, fname_hbi{i});
end

%% Posteriors

prob = cell(length(data),numel(idx));
auc = nan(length(data),numel(idx));
auc_conf = nan(length(data),numel(idx));
roc = cell(length(data),numel(idx));
loglik = nan(length(data),numel(idx));
bic = nan(length(data),numel(idx));
for hbi_idx = 1:size(idx,1)
    hbi_model = load(fname_hbi{hbi_idx});
    success_filter = all(success(:,idx(hbi_idx,:)),2);
    for model_idx = 1:size(idx,2)
        wrapper = str2func( ...
            sprintf('wrapper_post_%d', idx(hbi_idx,model_idx)));

        for j = 1:length(data)
            if ~success_filter(j), continue, end
            [loglik(j,((hbi_idx-1)*size(idx,2)+model_idx)), ...
                prob{j,((hbi_idx-1)*size(idx,2)+model_idx)}, ...
                choice] = wrapper( ...
                hbi_model.cbm.output.parameters{model_idx}( ...
                j - sum(~success_filter(1:j)),:), ...
                data{j});
            bic(j,((hbi_idx-1)*size(idx,2)+model_idx)) = ...
                -2*loglik(j,((hbi_idx-1)*size(idx,2)+model_idx)) + ...
                length(hbi_model.cbm.output.parameters{model_idx}( ...
                j - sum(~success_filter(1:j)),:)) * ...
                log(length(choice));
            roc{j,((hbi_idx-1)*size(idx,2)+model_idx)} = ...
                rocmetrics(choice, ...
                prob{j,((hbi_idx-1)*size(idx,2)+model_idx)}, [0,1], ...
                NumBootstraps=500,BootstrapOptions=statset(UseParallel=true));
            auc(j,((hbi_idx-1)*size(idx,2)+model_idx)) = ...
                roc{j,((hbi_idx-1)*size(idx,2)+model_idx)}.AUC(1,1);
            auc_conf(j,((hbi_idx-1)*size(idx,2)+model_idx)) = ...
                roc{j,((hbi_idx-1)*size(idx,2)+model_idx)}.AUC(3,1) - ...
                roc{j,((hbi_idx-1)*size(idx,2)+model_idx)}.AUC(2,1);
        end
    end
end
auc_matrix = reshape(mean(auc,"omitmissing"),size(idx'))';

for hbi_idx = 1:size(idx,1)
    load(fname_hbi{hbi_idx});
    cbm.output.auc = ...
        auc(:,((hbi_idx-1)*size(idx,2))+(1:size(idx,2)));
    cbm.output.auc_conf = ...
        auc_conf(:,((hbi_idx-1)*size(idx,2))+(1:size(idx,2)));
    cbm.output.loglik = ...
        loglik(:,((hbi_idx-1)*size(idx,2))+(1:size(idx,2)));
    cbm.output.bic = ...
        bic(:,((hbi_idx-1)*size(idx,2))+(1:size(idx,2)));
    save(fname_hbi{hbi_idx},"cbm");
end
