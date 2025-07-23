clc;
clear;
close all;

%% Step 1: Prepare All Data
fprintf('Loading and preparing data...\n');

% Load data from files
load('MTAERNetPlusv2_Final.mat', 'trainedNet', 'minVals', 'maxVals');
labels = readtable('labels.csv', 'VariableNamingRule', 'preserve');

% Define target variable names
targetNames = ["Anthocyanin", "TPC", "TFC", "DPPH"];
originalColNames = ["Anthocyanin (mg/100g)", "TPC (mg GAE/g)", "TFC (mg QE/g)", "DPPH % Inhibition"];

% Fix column names in min/max tables
minVals = renamevars(minVals, minVals.Properties.VariableNames, targetNames);
maxVals = renamevars(maxVals, maxVals.Properties.VariableNames, targetNames);

% Create grouping column from filenames
groupingColumn = 'stage'; 
image_filenames = labels.image;
new_column_data = cell(height(labels), 1);
for i = 1:height(labels)
    fname = lower(image_filenames(i));
    if contains(fname, 'stage1')
        new_column_data{i} = 'Stage 1';
    elseif contains(fname, 'stage2')
        new_column_data{i} = 'Stage 2';
    elseif contains(fname, 'stage3')
        new_column_data{i} = 'Stage 3';
    elseif contains(fname, 'stage4')
        new_column_data{i} = 'Stage 4';    
    else
        new_column_data{i} = 'Unknown';
    end
end
labels.(groupingColumn) = new_column_data; 
labels.(groupingColumn) = categorical(labels.(groupingColumn));

% Create simplified column names and file paths
labels.image = strtrim(string(labels.image));
labels.fullpath = fullfile(pwd, 'images', labels.image);
for i = 1:numel(targetNames)
    labels.(targetNames(i)) = labels.(originalColNames(i));
end

% Split data into validation set
rng(1);
labels = labels(randperm(height(labels)), :);
nTrain = round(0.8 * height(labels));
valData = labels(nTrain+1:end, :);

% Make predictions for all targets at once
inputSize = [224 224 3];
readFcn = @(x) preprocessImage(x, inputSize);
valImds = imageDatastore(valData.fullpath, 'ReadFcn', readFcn);
predictions_norm_raw = predict(trainedNet, valImds);
predictions_squeezed = squeeze(predictions_norm_raw);

fprintf('Data preparation complete.\n\n');

%% Step 2: Loop Through Targets and Generate Plots
fprintf('Starting to generate plots for all targets...\n');

for i = 1:numel(targetNames)
    generatePlotsForTarget(valData, predictions_squeezed, minVals, maxVals, i, targetNames(i));
end

fprintf('\nAll plots generated successfully.\n');

%% ========================================================================
%  LOCAL HELPER FUNCTION
%  (This must be at the end of the script file)
%  ========================================================================
function generatePlotsForTarget(valData, all_predictions_squeezed, minVals, maxVals, targetIndex, currentTarget)
    % This function generates the two required plots for a single target variable.

    fprintf('--- Processing: %s ---\n', currentTarget);
    
    % --- Part A: Denormalize predictions for the current target ---
    pred_norm_for_target = all_predictions_squeezed(targetIndex, :)';
    
    minVal = minVals.(currentTarget);
    maxVal = maxVals.(currentTarget);
    pred_denormalized = pred_norm_for_target * (maxVal - minVal) + minVal;
    
    ground_truth = valData.(currentTarget);
    
    % --- Part B: Generate Line Plot ---
    [group_ids, stage_names] = findgroups(valData.stage);
    mean_true_values = splitapply(@mean, ground_truth, group_ids);
    mean_predicted_values = splitapply(@mean, pred_denormalized, group_ids);

    % FIX: Use sprintf to create robust figure names and titles
    figure('Name', sprintf('Line Plot: %s', currentTarget));
    hold on;
    plot(stage_names, mean_true_values, 'g-o', 'LineWidth', 2, 'MarkerFaceColor', 'w', 'MarkerSize', 8);
    plot(stage_names, mean_predicted_values, 'b-^', 'LineWidth', 2, 'MarkerFaceColor', 'w', 'MarkerSize', 8);
    hold off;
    title(sprintf('Stage-wise %s: True vs MTAERNetPlusv2', currentTarget));
    xlabel('Developmental Stage');
    ylabel(char(currentTarget));
    legend('True', 'MTAERNetPlusv2', 'Location', 'northeast');
    grid on;
    box on;

    % --- Part C: Generate Bar Plot of Errors ---
    absolute_error = abs(ground_truth - pred_denormalized);
    mean_abs_error_by_group = splitapply(@mean, absolute_error, group_ids);
    
    figure('Name', sprintf('Error Bar Plot: %s', currentTarget));
    bar(stage_names, mean_abs_error_by_group);
    title(sprintf('Mean Absolute Prediction Error for %s by stage', currentTarget));
    xlabel('Stage');
    ylabel(sprintf('Mean Absolute Error (%s)', currentTarget));
    grid on;
end