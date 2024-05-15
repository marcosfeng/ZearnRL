rng(37909890)
% © 1998-2023 RANDOM.ORG
% Timestamp: 2023-10-04 18:38:28 UTC

% Add paths
addpath(fullfile('..','codes'));
addpath(fullfile('..', 'zearn_codes'));
addpath(fullfile('..', 'zearn_codes', 'ql_wrappers'));
addpath(fullfile('ql_subj_results'));

% Load prechosen models and common data
load("ranked_ql.mat");
fdata = load('../data/sample_data.mat');
data = fdata.data;

%% Re-estimate top models with more precision

models = cell(1, length(ranked_indices));
fname = cell(1, length(ranked_indices));

% Update the prior structure for Q-learning models
v = 6.25/2;
num_parameters = 5;
prior_ql = struct('mean', zeros(num_parameters, 1), 'variance', v);

for i = 1:length(ranked_indices)
    index = ranked_indices(i); % Get the model index
    models{i} = str2func(sprintf('wrapper_function_%d', index));
    fname{i} = strcat( ...
        sprintf('ql_refine/subj/refine_ql_%d_', index), ...
        '%d.mat');

    % Load previously saved results for mean updates
    loaded_data = load(sprintf('ql_subj_results/lap_ql_%d.mat', index));
    prior_ql.mean = prior_ql.mean + ...
        mean(loaded_data.cbm.output.parameters, 1)';
end
prior_ql.mean = prior_ql.mean / length(ranked_indices);

% PCONFIG structure with refined setup (2x default)
pconfig = struct();
pconfig.numinit = min(7 * num_parameters, 100) * 2;
pconfig.numinit_med = 70 * 2;
pconfig.numinit_up = 100 * 2;
pconfig.tolgrad = .001001 / 2;
pconfig.tolgrad_liberal = .1 / 2;
pconfig.prior_for_bads = 0;

num_subjects = length(data);
success = nan(num_subjects*length(ranked_indices),1);
% Refined estimation loop
parfor i = 1:(length(ranked_indices)*num_subjects)
    model_idx = floor((i-1)/num_subjects) + 1;
    subj_idx = mod(i-1, num_subjects) + 1;

    fname_subj = sprintf(fname{model_idx}, subj_idx);

    [~, success(i)] = ...
        cbm_lap(data(subj_idx), models{model_idx}, ...
        prior_ql, fname_subj, pconfig);
end
success = reshape(success,[num_subjects,length(ranked_indices)]);
success = logical(success);
save("ql_refine/success.mat","success");

%% Histograms by valid log evidence

load("ql_refine/success.mat");

fname_subjs = cell(num_subjects,length(models));
for m = 1:length(models)
    fname{m} = sprintf('refine_ql_%d', ranked_indices(m));
    % Aggregate results for each subject
    for subj = 1:num_subjects
        % Construct the filename for the current subject's results
        fname_subjs{subj,m} = sprintf(strcat('ql_refine/subj/', ...
            fname{m}, '_%d.mat'), subj);
    end
    fname{m} = strcat('ql_refine/', fname{m}, '.mat');
    cbm_lap_aggregate(fname_subjs(success(:,m),m),fname{m});
end

model_desc = cell(size(fname));
for i = 1:length(fname)
    % Create a cell array for the model descriptions
    % Extract the number from the filename
    num = regexp(fname{i}, '\d+', 'match');
    num = num{1}; % Assuming there's only one number in the filename
    
    % Construct the path to the corresponding .m file
    wrapper_filename = fullfile('..', 'zearn_codes', ...
        'ql_wrappers', ['wrapper_function_' num '.m']);
    
    % Read the .m file
    try
        file_contents = fileread(wrapper_filename);
    catch
        model_desc{i} = sprintf('Model %s: File not found', num);
        continue; % Skip this iteration if file not found
    end

    % Extract the outcome variable name
    outcome_match = regexp(file_contents, ...
        'subj\.outcome\s*=\s*subj\.(\w+);', 'tokens');
    % Extract the outcome variable name
    action_match  = regexp(file_contents, ...
        'subj\.action\s*=\s*subj\.(\w+);', 'tokens');

    if ~isempty(outcome_match) && ~isempty(action_match)
        outcome_var = outcome_match{1}{1};
        action_var  = action_match{1}{1};
        
        % Construct the model description
        model_desc{i} = sprintf('Action: %s, Outcome: %s', ...
            (action_var), (outcome_var));
    else
        % If the pattern is not found, use a placeholder
        model_desc{i} = 'Outcome and Actions not found in wrapper';
    end
end

% Convert the log evidence to non-scientific notation using a cell array
log_evidence = zeros(1, length(fname));
log_evidence_non_sci = cell(size(log_evidence));
for i = 1:length(log_evidence)
    loaded_data = load(fname{i});
    log_evidence(i) = ...
        sum(loaded_data.cbm.output.log_evidence);
    log_evidence_non_sci{i} = num2str(log_evidence(i), '%.2f');
end

% Create the table with the model description and log evidence
T = table(model_desc(:), log_evidence_non_sci(:), ...
    'VariableNames', {'Model', 'Log Evidence'});

% Display the table
disp(T);

%% Individual HBI to get BICs

% Rank the models by log evidence and get the indices of the top 4
[~, top4_indices] = sort(log_evidence, 'descend');
fname_hbi = compose( ...
    'ql_refine/hbi_ql%d.mat',ranked_indices);
pconfig = struct();
pconfig.maxiter = 300;
parfor i = 1:length(fname_hbi)
    cbm_hbi(data(success(:,i)), models(i), fname(i), fname_hbi{i}, pconfig);
end

%% Posteriors from top models
prob = cell(length(data),length(fname_hbi));
auc = nan(length(data),length(fname_hbi));
auc_conf = nan(length(data),length(fname_hbi));
roc = cell(length(data),length(fname_hbi));
loglik = nan(length(data),length(fname_hbi));
for i = 1:length(fname_hbi)
    hbi_model = load(fname_hbi{i});
    % hbi_model = load(fname{i});
    filtered_data = data(success(:,i));
    wrapper = str2func(sprintf('wrapper_post_%d', ranked_indices(i)));
    for j = 1:length(filtered_data)
        [loglik(j,i), prob{j,i}, choice, q_values] = wrapper( ...
            hbi_model.cbm.output.parameters{1, 1}(j,:), ...
            filtered_data{j});
        % roc{i,j} = rocmetrics(choice,prob{i,j},[0,1], ...
        %     NumBootstraps=100,BootstrapOptions=statset(UseParallel=true), ...
        %     BootstrapType="student",NumBootstrapsStudentizedSE=100);
        roc{j,i} = rocmetrics(choice,prob{j,i},[0,1], ...
            NumBootstraps=500,BootstrapOptions=statset(UseParallel=true));
        auc(j,i) = roc{j,i}.AUC(1,1);
        auc_conf(j,i) = roc{j,i}.AUC(3,1) - roc{j,i}.AUC(2,1);
    end
end

% auc_weights = 1./(auc_conf);
% auc_weights(isinf(auc_weights) | isnan(auc_weights)) = 0;
% mean(auc, 2, Weights=auc_weights);
for i = 1:length(fname_hbi)
    load(fname_hbi{i});
    cbm.output.auc = auc(:,i);
    cbm.output.auc_conf = auc_conf(:,i);
    cbm.output.loglik = loglik(:,i);
    save(fname_hbi{i},"cbm");
end

%% Run cbm_hbi for top models

auc = nan(length(data),length(fname_hbi));
auc_conf = nan(length(data),length(fname_hbi));
loglik = nan(length(data),length(fname_hbi));
for i = 1:length(fname_hbi)
    load(fname_hbi{i});
    auc(:,i) = cbm.output.auc;
    auc_conf(:,i) = cbm.output.auc_conf;
    loglik(:,i) = cbm.output.loglik;
end

% Top AUC
[~, top_auc] = sort(mean(auc), 'descend');
% Smallest BIC
[~, top_avgbic] = sort( ...
    (num_parameters*log(length(data))-2*sum(loglik))/length(data), ...
    'ascend');
% Check data is the same size:
% sum(success(:,top_auc(1))) == sum(success(:,top_avgbic(1)))

fname_hbi = 'ql_refine/hbi_topQL.mat';
pconfig = struct();
pconfig.maxiter = 500;
cbm_hbi(data, models([top_auc(1),top_avgbic(1)]), ...
    fname([top_auc(1),top_avgbic(1)]), fname_hbi, pconfig);
cbm_hbi_null(data, fname_hbi);

% Load the HBI results and display them
fname_hbi_loaded = load(fname_hbi);
hbi_results = fname_hbi_loaded.cbm;
hbi_results.output
