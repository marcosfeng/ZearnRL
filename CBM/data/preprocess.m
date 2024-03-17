% Combine individual data into one struct
Nsubj = 295;
% Initialize an empty cell array to hold the individual subject data
data = cell(Nsubj, 1);
    
% Loop through each subject's .mat file
for i = 1:Nsubj
    % Create the full path to the individual .mat file
    filename = sprintf('individual/subj_%d.mat', i);
    
    % Check if the file exists
    if isfile(filename)
        % Load the individual .mat file
        individual_data = load(filename);
        
        % Add this struct to the cell array
        data{i, 1} = individual_data;
    else
        fprintf('File %s does not exist.\n', filename);
    end
end

% Save the cell array to a new .mat file within the same folder
save(sprintf('all_data.mat'), 'data');
% rmdir("individual/","s")