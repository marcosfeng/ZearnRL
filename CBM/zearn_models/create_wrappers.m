state_vars = {'activest', 'minutes', 'badges', 'boosts', 'alerts'};
all_combinations = {};

for i = 1:length(state_vars)
    % Randomly select an outcome variable
    outcome_var = state_vars{i};
    % Remove the outcome variable from state_vars
    states = setdiff(state_vars, outcome_var);
    
    for j = 1:length(states)
        combs = nchoosek(states, j);
        
        % Add outcome_var to the first column of each combination
        combs_with_outcome = cellfun(@(x) [outcome_var, x], mat2cell(combs, ones(size(combs, 1), 1), size(combs, 2)), 'UniformOutput', false);
        
        all_combinations = [all_combinations; combs_with_outcome];
    end
end

for i = 1:length(all_combinations)
    fname = sprintf('wrappers/wrapper_function_%d.m', i);
    fid = fopen(fname, 'w');
    
    fprintf(fid, 'function [loglik] = wrapper_function_%d(parameters, subj)\n', i);
    
    comb = all_combinations{i};
    outcome_var = comb{1};  % Extract the outcome variable from the first element
    
    fprintf(fid, '    %% Outcome variable: %s\n', outcome_var);
    fprintf(fid, '    subj.outcome = subj.%s;\n', outcome_var);
    
    state_vars = comb(2:end);  % Extract the state variables from the remaining elements
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
