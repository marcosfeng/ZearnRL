% Initialize cell arrays for data, models, and fcbm_maps
all_data = cell(1, 75);
models = cell(1, 75);
fcbm_maps = cell(1, 75);

% Loop through each dataset
for i = 1:75
    % Load the data for this dataset
    fdata = load(sprintf('../data/individual/data_%d/all_data.mat', i));
    data  = fdata.data;
    
    % Store the data, model, and fcbm_map for this dataset
    all_data{i} = data;
    models{i} = @actor_critic_model;
    fcbm_maps{i} = sprintf('lap_ac_%d.mat', i);
end

% Run cbm_hbi for model comparison
fname_hbi = 'hbi_model_comparison.mat';
cbm_hbi(all_data, models, fcbm_maps, fname_hbi);
