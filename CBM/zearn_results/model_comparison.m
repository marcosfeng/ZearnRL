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
    % 'aggr_results/lap_aggr_logit_model_2.mat', ... % Logit
    'aggr_results/lap_aggr_q_model_2.mat'}; % Q-learning
models = {
    @posterior_baseline_model_2, ...
    % @posterior_logit_model_2, ...
    @posterior_q_model_2};
num_parameters = [
    % 1, 5, 5
    1, 5
    ];

%% Re-estimate all relevant models

fname_template = {
    'comp_results/lap_baseline.mat', ... % Baseline
    % 'comp_results/lap_logit.mat', ...  % Logit
    'comp_results/lap_ql.mat'}; % Q-learning

% Define the prior variance
v = 6.25;
num_subjects = length(data);
priors = struct([]);
for i = 1:length(num_parameters)
    priors{i} = struct('mean', zeros(num_parameters(i), 1), ...
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

parfor i = 1:length(num_parameters)
    % Run the cbm_lap function for the current model and subject
    [~, success(i)] = ...
        cbm_lap(data, models{i}, ...
        priors{i}, fname_template{i}, pconfig);
end

%% HBI

fname_hbi = {'comp_results/hbi_compare.mat'};

idx = [1,2];
% Pre-allocate structures to store aggregated results
fname_subjs = cell(size(idx,1));
pconfig = struct();
pconfig.maxiter = 50 * mult;
pconfig.tolL = 0.6931 / mult;
pconfig.tolx = 0.0100 / mult;

% cbm_bl = load(fname_template{1}).cbm;
% cbm_ql = load(fname_template{2}).cbm;
% success = cbm_ql.output.log_evidence > cbm_bl.output.log_evidence;
% 
% cbm_bl.profile.optim.flag = cbm_bl.profile.optim.flag(success);
% cbm_bl.profile.optim.gradient = cbm_bl.profile.optim.gradient(success);
% cbm_bl.math.A = cbm_bl.math.A(success);
% cbm_bl.math.Ainvdiag = cbm_bl.math.Ainvdiag(success);
% cbm_bl.math.lme = cbm_bl.math.lme(success);
% cbm_bl.math.logdetA = cbm_bl.math.logdetA(success);
% cbm_bl.math.loglik = cbm_bl.math.loglik(success);
% cbm_bl.math.theta = cbm_bl.math.theta(success);
% cbm_bl.output.log_evidence = cbm_bl.output.log_evidence(success);
% cbm_bl.output.parameters = cbm_bl.output.parameters(success);
% 
% cbm_ql.profile.optim.flag = cbm_ql.profile.optim.flag(success);
% cbm_ql.profile.optim.gradient = cbm_ql.profile.optim.gradient(success);
% cbm_ql.math.A = cbm_ql.math.A(success);
% cbm_ql.math.Ainvdiag = cbm_ql.math.Ainvdiag(success);
% cbm_ql.math.lme = cbm_ql.math.lme(success);
% cbm_ql.math.logdetA = cbm_ql.math.logdetA(success);
% cbm_ql.math.loglik = cbm_ql.math.loglik(success);
% cbm_ql.math.theta = cbm_ql.math.theta(success);
% cbm_ql.output.log_evidence = cbm_ql.output.log_evidence(success);
% cbm_ql.output.parameters = cbm_ql.output.parameters(success);
% 
% fname_template = {
%     'test/lap_baseline.mat', ... % Baseline
%     % 'comp_results/lap_logit.mat', ...  % Logit
%     'test/lap_ql.mat'}; % Q-learning
% cbm = cbm_bl;
% save(fname_template{1},"cbm");
% cbm = cbm_ql;
% save(fname_template{2},"cbm");

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
