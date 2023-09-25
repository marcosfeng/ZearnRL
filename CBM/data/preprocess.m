% Initialize an empty cell array to hold the individual subject data
data = cell(210, 1);  % Assuming you have 210 subjects
    
% Loop through each subject's .mat file
for i = 1:210  % Replace 210 with the actual number of subjects if different
    % Create the full path to the individual .mat file
    filename = sprintf('individual/subj_%d.mat', i);
    
    % Check if the file exists
    if isfile(filename)
        % Load the individual .mat file
        individual_data = load(filename);
        
        % Create a struct to hold the loaded data
        subj_struct = struct('actions', individual_data.actions, ...
                             'activest', individual_data.activest, ...
                             'minutes', individual_data.minutes, ...
                             'badges', individual_data.badges, ...
                             'boosts', individual_data.boosts, ...
                             'alerts', individual_data.alerts, ...
                             'simmed', individual_data.simmed);
        
        % Add this struct to the cell array
        data{i, 1} = subj_struct;
    else
        fprintf('File %s does not exist.\n', filename);
    end
end

% Save the cell array to a new .mat file within the same folder
save(sprintf('all_data.mat'), 'data');