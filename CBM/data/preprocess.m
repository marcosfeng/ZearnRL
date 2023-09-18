% Initialize an empty cell array to hold the individual subject data
data = cell(210, 1);  % Assuming you have 210 subjects

% Loop through each subject's .mat file
for i = 1:210  % Replace 210 with the actual number of subjects if different
    % Load the individual .mat file
    filename = sprintf('individual/subj_%d.mat', i);
    individual_data = load(filename);
    
    % Create a struct to hold the loaded data
    subj_struct = struct('actions', individual_data.actions, ...
                         'outcome', individual_data.outcome, ...
                         'simmed', individual_data.simmed);
    
    % Add this struct to the cell array
    data{i, 1} = subj_struct;
end

% Save the cell array to a new .mat file
save('all_data.mat', 'data');