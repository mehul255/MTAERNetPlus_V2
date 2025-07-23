% GradCAM_MultiOutput.m — Fixed version using dlnetwork and forward()
clc; clear; close all;

%% USER SETTING: Select validation image index
imageIdx = 85;  % Change this to visualize other images

%% Step 1: Load and preprocess validation image
labels = readtable('labels.csv', 'VariableNamingRule','preserve');
labels.image = strtrim(string(labels.image));
imageFolder = fullfile(pwd, 'images');
labels.fullpath = fullfile(imageFolder, labels.image);
labels = labels(isfile(labels.fullpath), :);

% Validation split
rng(1);
labels = labels(randperm(height(labels)), :);
nTrain = round(0.8 * height(labels));
valData = labels(nTrain+1:end, :);
valImds = imageDatastore(valData.fullpath);

% Load image
img = im2single(imresize(imread(valImds.Files{imageIdx}), [224 224]));
dlImg = dlarray(img, 'SSC');  % spatial-spatial-channel

%% Step 2: Rebuild ResNet18 + Multioutput Head (no regression layer)
net = resnet18;
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'});

% Regression head
regressionHead = [
    fullyConnectedLayer(128,'Name','fc_shared')
    reluLayer('Name','relu_shared')
    fullyConnectedLayer(64,'Name','fc_regress')
    reluLayer('Name','relu_regress')
    dropoutLayer(0.3,'Name','drop')
    fullyConnectedLayer(4,'Name','fc_final')  % no regressionLayer here
];

lgraph = addLayers(lgraph, regressionHead);
lgraph = connectLayers(lgraph,'pool5','fc_shared');

% Convert to dlnetwork
dlnet = dlnetwork(lgraph);

%% Step 3: Grad-CAM for 'res5b_relu'
targetLayer = 'res5b_relu';

% Forward pass to extract activations at target layer
featureMapStruct = forward(dlnet, dlImg, 'Outputs', targetLayer);
featureMap = extractdata(featureMapStruct);  % shape: [H W C]

% Global Average Pooling → Grad-CAM weights
weights = mean(featureMap, [1 2]);  % 1 × 1 × C

% Weighted sum across channels
cMap = sum(featureMap .* reshape(weights, 1, 1, []), 3);
cMap = rescale(cMap);

%% Step 4: Resize CAM and Overlay on Original Image
cMap = imresize(cMap, [size(img,1), size(img,2)]);  % Resize to match image

figure;
imshow(img);  % Original image
hold on;

% Overlay Grad-CAM heatmap
h = imagesc(cMap);
colormap jet;
colorbar;
h.AlphaData = 0.5;  % Transparency of overlay
title(sprintf('Grad-CAM Overlay — Validation Image #%d', imageIdx));
