state_vars = {'NNDSVD_student1', ...
    'NNDSVD_student2', ...
    'NNDSVD_student3', ...
    'NNDSVD_student4'};
action_vars = { ...
    % 'NNDSVD_teacher1', ...
    'NNDSVD_teacher2', ...
    'NNDSVD_teacher3', ...
    % 'NNDSVD_teacher4'
    };

%% Q-learning

for i = 1:length(state_vars)
    for j = 1:length(action_vars)
        fname = sprintf('ql_wrappers/wrapper_function_%d.m', ...
            ((i-1) * length(action_vars) + j));
        fid = fopen(fname, 'w');
    
        fprintf(fid, 'function [loglik] = wrapper_function_%d(parameters, subj)\n', ...
            ((i-1) * length(action_vars) + j));
    
        fprintf(fid, '    %% Outcome variable: %s\n', state_vars{i});
        fprintf(fid, '    subj.outcome = subj.%s;\n', state_vars{i});
        fprintf(fid, '    %% Action variable: %s\n', action_vars{j});
        fprintf(fid, '    subj.action = subj.%s;\n', action_vars{j});
    
        fprintf(fid, '    loglik = q_model(parameters, subj);\n');
        fprintf(fid, 'end\n');
        
        fclose(fid);
    end
end

%% Actor-Critic
all_combinations = {};

for i = 1:length(state_vars)
    % Select an outcome variable
    outcome_var = state_vars{i};
    % Remove the outcome variable from state_vars
    states = setdiff(state_vars, outcome_var);
    
    for j = 1:length(states)
        combs = nchoosek(states, j);
        
        for k = 1:length(action_vars)  % Loop over action variables
            action_var = action_vars{k};

            combs_full = cellfun(@(x) [outcome_var, x, action_var], ...
                mat2cell(combs, ones(size(combs, 1), 1), ...
                size(combs, 2)), 'UniformOutput', false);
            
            all_combinations = [all_combinations; combs_full];
        end
    end
end

for i = 1:length(all_combinations)
    fname = sprintf('ac_wrappers/wrapper_function_%d.m', i);
    fid = fopen(fname, 'w');
    
    fprintf(fid, 'function [loglik] = wrapper_function_%d(parameters, subj)\n', i);
    
    comb = all_combinations{i};
    outcome_var = comb{1};  % Extract the outcome variable
    action_var = comb{end}; % Extract the action variable
    
    fprintf(fid, '    %% Outcome variable: %s\n', outcome_var);
    fprintf(fid, '    subj.outcome = subj.%s;\n', outcome_var);
    fprintf(fid, '    %% Action variable: %s\n', action_var);
    fprintf(fid, '    subj.action = subj.%s;\n', action_var);
    
    state_vars = comb(2:end-1);  % Extract the state variables
    fprintf(fid, '    %% State variables: ');
    for j = 1:length(state_vars)
        fprintf(fid, '%s ', state_vars{j});
    end
    fprintf(fid, '\n');
    
    fprintf(fid, '    subj.state = [');
    for j = 1:length(state_vars)
        fprintf(fid, 'subj.%s ', state_vars{j});
        if j < length(state_vars)
            fprintf(fid, ', ');
        end
    end
    fprintf(fid, '];\n');
    
    fprintf(fid, '    loglik = actor_critic_model(parameters, subj);\n');
    fprintf(fid, 'end\n');
    
    fclose(fid);
end
