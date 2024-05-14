rng(37909890)
% © 1998-2023 RANDOM.ORG
% Timestamp: 2023-10-04 18:38:28 UTC

% Add paths
addpath(fullfile('..','codes'));
addpath(fullfile('..','zearn_codes'));
addpath(fullfile('..','zearn_codes','ac_wrappers'));
addpath(fullfile('ac_subj_results'));

% Load the common data for all datasets
load("top10_ac.mat");
fdata = load('../data/sample_data.mat');
data  = fdata.data;

%% Re-estimate top models with more precision

models = cell(1,10);
fname = cell(1,10);
priors = cell(1,10);

% Create the prior structure for your new model
v = 6.25/2;
max_num_parameters = 0;
for i = 1:10
    models{i} = str2func(sprintf('wrapper_function_%d', top10_indices(i)));
    fname{i} = sprintf('ac_subj_results/lap_ac_%d.mat', top10_indices(i));
    
    % Determine the number of parameters in your model.
    % Read the contents of the wrapper function file
    file_contents = fileread( ...
        sprintf('wrapper_function_%d.m', top10_indices(i)));
    % Find the subj.state assignment line
    state_line = regexp(file_contents, ...
        'subj\.state\s*=\s*\[.*?\]', 'match', 'once');
    % Extract the elements inside the square brackets
    elements = regexp(state_line, 'subj\.\w+', 'match');
    % Count the number of parameters
    num_states = numel(elements);
    num_parameters = 5 + 2*num_states;
    max_num_parameters = max(max_num_parameters, num_parameters);

    % Create the prior structure from previous estimation
    priors{i} = struct('mean', zeros(num_parameters, 1), 'variance', v);

    loaded_data = load(fname{i});
    priors{i}.mean = mean(loaded_data.cbm.output.parameters,1)';
    fname{i} = sprintf('ac_refine/subj/refine_ac_%d', top10_indices(i));
    fname{i} = strcat(fname{i}, '_%d.mat');
end

% PCONFIG structure with refined setup (2x default)
pconfig = struct();
pconfig.numinit = min(7 * max_num_parameters, 100) * 2;
pconfig.numinit_med = 70 * 2;
pconfig.numinit_up = 100 * 2;
pconfig.tolgrad = .001001 / 2;
pconfig.tolgrad_liberal = .1 / 2;
pconfig.prior_for_bads = 0;

num_subjects = length(data);
success = nan(num_subjects*length(models),1);
parfor i = 1:(length(models)*num_subjects)
    model_idx = floor((i-1)/num_subjects) + 1;
    subj_idx = mod(i-1, num_subjects) + 1;

    % Construct filename for saving output
    fname_subj = sprintf(fname{model_idx}, subj_idx);
    % % if you need to re-run models
    % if exist(fname_subj,"file") == 2
    %     continue;
    % end

    % Run the cbm_lap function for the current model and subject
    [~, success(i)] = ...
        cbm_lap(data(subj_idx), models{model_idx}, ...
        priors{model_idx}, fname_subj, pconfig);
end
success = reshape(success,[num_subjects,length(models)]);
success = logical(success);
save("ac_refine/success.mat","success");

%% Histograms by valid log evidence

load("ac_refine/success.mat");

fname_subjs = cell(num_subjects,length(models));
for m = 1:length(models)
    fname{m} = sprintf('refine_ac_%d', top10_indices(m));
    % Aggregate results for each subject
    for subj = 1:num_subjects
        % Construct the filename for the current subject's results
        fname_subjs{subj,m} = sprintf(strcat('ac_refine/subj/', ...
            fname{m}, '_%d.mat'), subj);
    end
    fname{m} = strcat('ac_refine/', fname{m}, '.mat');
    cbm_lap_aggregate(fname_subjs(success(:,m),m),fname{m});
end

model_desc = cell(size(fname));
for i = 1:length(fname)
    % 2) Create a cell array for the model descriptions
    % Extract the number from the filename
    num = regexp(fname{i}, '\d+', 'match');
    num = num{1}; % Assuming there's only one number in the filename
    
    % Construct the path to the corresponding .m file
    wrapper_filename = fullfile('..', 'zearn_codes', ...
        'ac_wrappers', ['wrapper_function_' num '.m']);
    
    % Read the .m file
    file_contents = fileread(wrapper_filename);
    
    % Extract the outcome variable name
    outcome_match = regexp(file_contents, ...
        'subj\.outcome\s*=\s*subj\.(\w+);', 'tokens');
    % Extract the outcome variable name
    action_match  = regexp(file_contents, ...
        'subj\.action\s*=\s*subj\.(\w+);', 'tokens');
    % Extract the entire string of state variables
    state_match   = regexp(file_contents, ...
        'subj\.state\s*=\s*\[(subj\.\w+\s*(?:,\s*subj\.\w+\s*)*)\];', 'match');
    
    if ~isempty(outcome_match) && ~isempty(state_match)
        outcome_var = outcome_match{1}{1};
        action_var  = action_match{1}{1};
        state_var_string = state_match{1};
        
        % Now split the state variable string into individual variables
        state_vars = regexp(state_var_string, 'subj\.(\w+)', 'tokens');
        % Flatten the nested cell array resulting from regexp
        state_vars = [state_vars{2:length(state_vars)}];
        
        % Construct the model description
        model_desc{i} = sprintf('Action: %s, Outcome: %s, States: %s', ...
            (action_var), (outcome_var), strjoin((state_vars), ', '));
    else
        % If the pattern is not found, use a placeholder
        model_desc{i} = 'Outcome and States not found in wrapper';
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

% Rank the models by log evidence and get the indices of the top 5
[~, top5_indices] = sort(log_evidence ./ sum(success), 'descend');
top5_indices = top5_indices(1:5);

fname_hbi = compose( ...
    'ac_refine/hbi_ac%d.mat',top10_indices(top5_indices));
pconfig = struct();
pconfig.maxiter = 300;
parfor i = 1:length(fname_hbi)
    cbm_hbi(data(success(:,top5_indices(i))), models(top5_indices(i)), ...
        fname(top5_indices(i)), fname_hbi{i}, pconfig);
end

%% Posteriors from top models
prob = struct([]);
auc = nan(length(data),length(fname_hbi));
auc_conf = nan(length(data),length(fname_hbi));
roc = struct([]);
roc{1,1} = [0,1];
loglik = nan(length(data),length(fname_hbi));
for i = 1:length(fname_hbi)
    hbi_model = load(fname_hbi{i});
    % hbi_model = load(fname{i});
    wrapper = str2func( ...
        sprintf('wrapper_post_%d', top10_indices(top5_indices(i))));
    for j = 1:length(data)
        if ~success(j,top5_indices(i)), continue, end
        [loglik(j,i), prob{j,i}, choice, theta, w] = wrapper( ...
            hbi_model.cbm.output.parameters{1, 1}( ...
            j - sum(~success(1:j,top5_indices(i))),:), ...
            data{j});
        roc{j,i} = rocmetrics(choice,prob{j,i},[0,1], ...
            NumBootstraps=500,BootstrapOptions=statset(UseParallel=true));
        auc(j,i) = roc{j,i}.AUC(1,1);
        auc_conf(j,i) = roc{j,i}.AUC(3,1) - roc{j,i}.AUC(2,1);
    end
end

% auc_weights = 1./(auc_conf);
% auc_weights(isinf(auc_weights) | isnan(auc_weights)) = 0;
% mean(auc, 2, Weights=auc_weights);

%% Run cbm_hbi for top models

% Top AUC
[~, top_auc] = sort(mean(auc,"omitmissing"), 'descend'); % Get largest AUC
num_parameters = nan(length(fname_hbi),1);
for i = 1:length(fname_hbi)
    num_parameters(i) = length(priors{top5_indices(i)}.mean);
end
% Smallest BIC
[~, top_avgbic] = sort( ...
    (num_parameters' .* log(sum(success(:,top5_indices))) - ...
    2*sum(loglik,"omitmissing")) ./ sum(success(:,top5_indices)), ...
    'ascend'); 
% Check data is the same size:
% sum(success(:,top5_indices(top_auc(1)))) == ...
%     sum(success(:,top5_indices(top_avgbic(1))))
success_hbi = success(:,top5_indices(top_auc(1))) & ...
    success(:,top5_indices(top_avgbic(1)));

fname_subjs = cell(sum(success_hbi),2);
fname = compose('refine_ac_%d', ...
    top10_indices([top5_indices(top_auc(1)),top5_indices(top_avgbic(1))]));
fname_cbm_hbi = strcat('ac_refine/', ...
    compose('cmb_forHBI_%d', ...
    top10_indices([top5_indices(top_auc(1)),top5_indices(top_avgbic(1))])), ...
    '.mat');
for m = 1:2
    % Aggregate results for each subject
    for subj = 1:num_subjects
        if ~success_hbi(subj), continue, end
        % Construct the filename for the current subject's results
        fname_subjs{subj-sum(~success_hbi(1:subj)),m} = ...
            sprintf(strcat('ac_refine/subj/', ...
            fname{m}, '_%d.mat'), subj);
    end
    cbm_lap_aggregate(fname_subjs(:,m),fname_cbm_hbi{m});
end

fname_hbi = 'ac_refine/hbi_topAC.mat';
pconfig = struct();
pconfig.maxiter = 500;
cbm_hbi(data(success_hbi), ...
    models(top5_indices([top_auc(1),top_avgbic(1)])), ...
    fname_cbm_hbi, fname_hbi, pconfig);
cbm_hbi_null(data(success_hbi), fname_hbi);
