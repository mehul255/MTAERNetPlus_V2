
% CREATE COMMON TRAINING-VALIDATION SPLIT
% For ResNet18 and CustomCNN
% MATLAB R2023b

clc;
clear;

%% Step 1: Load dataset

labels = readtable('labels.csv');

labels.image = strtrim(string(labels.image));

imageFolder = fullfile(pwd, 'images');

labels.fullpath = fullfile(imageFolder, labels.image);

% Retain only existing images with valid anthocyanin values
validRows = isfile(labels.fullpath) & ...
    ~isnan(labels.anthocyanin);

labels = labels(validRows, :);

%% Step 2: Fix random seed

rng(1);

numImages = height(labels);

%% Step 3: Generate common 80:20 split

randomOrder = randperm(numImages);

nTrain = round(0.80 * numImages);

trainIdx = randomOrder(1:nTrain);

valIdx = randomOrder(nTrain+1:end);

trainData = labels(trainIdx, :);

valData = labels(valIdx, :);

%% Step 4: Display split information

fprintf('\nCOMMON DATA SPLIT\n');

fprintf('Total images: %d\n', numImages);

fprintf('Training images: %d\n', height(trainData));

fprintf('Validation images: %d\n', height(valData));

fprintf('\nTraining anthocyanin range: %.2f to %.2f\n', ...
    min(trainData.anthocyanin), ...
    max(trainData.anthocyanin));

fprintf('Validation anthocyanin range: %.2f to %.2f\n', ...
    min(valData.anthocyanin), ...
    max(valData.anthocyanin));

%% Step 5: Save common split

save('Common_Data_Split.mat', ...
    'labels', 'trainData', 'valData', ...
    'trainIdx', 'valIdx');

writetable(trainData, 'Common_Training_Data.xlsx');

writetable(valData, 'Common_Validation_Data.xlsx');

fprintf('\nCommon split saved successfully!\n');
