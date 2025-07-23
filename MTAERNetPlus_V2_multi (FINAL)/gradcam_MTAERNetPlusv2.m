clc; clear; close all;

%% Load model and label ranges
load('MTAERNetPlusv2_Final.mat', 'dlnet', 'minVals', 'maxVals');
targetNames = ["Anthocyanin", "TPC", "TFC", "DPPH"];

%% Select which output to visualize
[idx, tf] = listdlg('PromptString','Select output to visualize:', ...
    'ListString', cellstr(targetNames), 'SelectionMode','single');
if ~tf, disp('Cancelled'); return; end
outputIndex = idx;
traitName = targetNames(outputIndex);

%% Load image
[filename, pathname] = uigetfile({'*.jpg;*.png'}, 'Select image');
if isequal(filename, 0), disp('Cancelled'); return; end
img = imread(fullfile(pathname, filename));
img = im2double(imresize(img, dlnet.Layers(1).InputSize(1:2)));

% Optional contrast enhancement
for i = 1:3
    img(:,:,i) = adapthisteq(img(:,:,i));
end

dlImg = dlarray(single(img), 'SSC');

%% Choose feature layer
targetLayer = "res3b_relu";  % try also "shared_relu", "res5b_relu", etc.

%% Compute Grad-CAM
[camMap, scoreNorm] = dlfeval(@gradCAM_MTAER, dlnet, dlImg, targetLayer, outputIndex, minVals, maxVals);

%% Overlay heatmap
cmap = hot; % vivid colormap
heat = ind2rgb(uint8(camMap * 255), cmap);
overlay = 0.6 * heat + 0.4 * img;

figure;
imshow(overlay);
colorbar;
title(sprintf("Grad-CAM: %s (%.2f)", traitName, scoreNorm));

%% Function
function [camMap, scoreNorm] = gradCAM_MTAER(dlnet, dlImg, targetLayer, outputIndex, minVals, maxVals)
    % Get feature maps
    features = forward(dlnet, dlImg, 'Outputs', targetLayer);

    % Get all 4 outputs
    out1 = forward(dlnet, dlImg, 'Outputs', 'out_b1');
    out2 = forward(dlnet, dlImg, 'Outputs', 'out_b2');
    out3 = forward(dlnet, dlImg, 'Outputs', 'out_b3');
    out4 = forward(dlnet, dlImg, 'Outputs', 'out_b4');

    % Select score
    scores = [out1, out2, out3, out4];
    score = scores(outputIndex);

    % Denormalize
    traitName = ["Anthocyanin","TPC","TFC","DPPH"];
    minVal = minVals{1, "min_" + traitName(outputIndex)};
    maxVal = maxVals{1, "max_" + traitName(outputIndex)};
    scoreNorm = score * (maxVal - minVal) + minVal;

    % Compute gradient
    gradients = dlgradient(score, features);
    pooledGrad = mean(gradients, [1 2]);
    camMap = sum(features .* pooledGrad, 3);
    camMap = max(camMap, 0);

    % Resize and enhance
    camMap = extractdata(camMap);
    camMap = imresize(camMap, dlnet.Layers(1).InputSize(1:2));
    camMap = imsharpen(camMap);
    camMap = mat2gray(camMap);
    camMap = imadjust(camMap);
    camMap(camMap < 0.3) = 0;
end
