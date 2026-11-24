
clc;
clear;
close all;

%% STEP 1: LOAD TRAINED MODEL AND NORMALIZATION VALUES

load('MTAERNetPlusv2_Final.mat', ...
    'dlnet', 'minVals', 'maxVals');

targetNames = ["Anthocyanin", "TPC", "TFC", "DPPH"];

outputLayers = ["out_b1", "out_b2", "out_b3", "out_b4"];

%% STEP 2: SELECT TARGET OUTPUT

[idx, tf] = listdlg( ...
    'PromptString', 'Select biochemical parameter:', ...
    'ListString', cellstr(targetNames), ...
    'SelectionMode', 'single');

if ~tf
    disp('Operation cancelled.');
    return;
end

outputIndex = idx;

traitName = targetNames(outputIndex);

outputLayer = outputLayers(outputIndex);

fprintf('\nSelected output: %s\n', traitName);

%% STEP 3: SELECT IMAGE

[filename, pathname] = uigetfile( ...
    {'*.jpg;*.jpeg;*.png;*.tif', 'Image Files'}, ...
    'Select Delonix regia flower image');

if isequal(filename, 0)
    disp('Image selection cancelled.');
    return;
end

imagePath = fullfile(pathname, filename);

img = imread(imagePath);

%% STEP 4: IMAGE PREPROCESSING

inputSize = dlnet.Layers(1).InputSize(1:2);

% Convert grayscale to RGB
if size(img, 3) == 1
    img = repmat(img, [1 1 3]);
end

% Remove extra channels
if size(img, 3) > 3
    img = img(:, :, 1:3);
end

% Resize image
img = im2double(imresize(img, inputSize));

% IMPORTANT:
% Keep this setting consistent with the preprocessing
% used during the original model training.

applyCLAHE = true;

if applyCLAHE
    for k = 1:3
        img(:, :, k) = adapthisteq(img(:, :, k));
    end
end

% Prepare dlarray with spatial, channel and batch dimensions
dlImg = dlarray(single(reshape(img, ...
    [size(img,1), size(img,2), 3, 1])), 'SSCB');

%% STEP 5: SELECT FEATURE LAYER

% This must be an upstream feature layer connected
% to the selected prediction branch.

targetLayer = "res3b_relu";

fprintf('Feature layer: %s\n', targetLayer);


%% DIAGNOSTIC: CHECK MODEL PREDICTION

% Check target normalization limits
traitNames = ["Anthocyanin", "TPC", "TFC", "DPPH"];

minVal = minVals{1, "min_" + traitNames(outputIndex)};
maxVal = maxVals{1, "max_" + traitNames(outputIndex)};

% Get selected output directly
rawOutput = predict(dlnet, dlImg, ...
    'Outputs', char(outputLayer));

rawOutput = double(gather(extractdata(rawOutput)));

% Convert to original units
predictedValue = rawOutput .* (maxVal - minVal) + minVal;

fprintf('\n========== PREDICTION CHECK ==========\n');

fprintf('Selected target: %s\n', traitName);
fprintf('Normalized output: %.6f\n', rawOutput);
fprintf('Minimum training value: %.4f\n', minVal);
fprintf('Maximum training value: %.4f\n', maxVal);
fprintf('Predicted value: %.4f\n', predictedValue);

fprintf('=====================================\n');

%% STEP 6: COMPUTE TARGET-SPECIFIC GRAD-CAM

[camMap, predictedValue] = dlfeval( ...
    @gradCAM_MTAER, ...
    dlnet, ...
    dlImg, ...
    targetLayer, ...
    outputLayer, ...
    outputIndex, ...
    minVals, ...
    maxVals);

fprintf('\n====================================\n');
fprintf('GRAD-CAM PREDICTION\n');
fprintf('====================================\n');

fprintf('%s prediction: %.4f\n', ...
    traitName, predictedValue);

%% STEP 7: DISPLAY ORIGINAL IMAGE

figure('Name', 'Original Image');

imshow(img);

title('Original Image');

%% STEP 8: DISPLAY NORMALIZED GRAD-CAM

figure('Name', 'Normalized Grad-CAM');

imagesc(camMap);

axis image;
axis off;

colormap(jet);

clim([0 1]);

cb = colorbar;

cb.Label.String = 'Normalized attribution intensity';

title(sprintf('Grad-CAM: %s', traitName));

%% STEP 9: DISPLAY GRAD-CAM OVERLAY

figure('Name', 'Grad-CAM Overlay');

imshow(img);

hold on;

h = imagesc(camMap);

% Transparency proportional to activation intensity
set(h, 'AlphaData', 0.45 .* camMap);

colormap(gca, jet);

clim([0 1]);

cb = colorbar;

cb.Label.String = 'Normalized attribution intensity';

title(sprintf('Grad-CAM: %s (%.2f)', ...
    traitName, predictedValue));

axis image;
axis off;

%% STEP 10: SAVE RESULTS

[saveName, savePath] = uiputfile( ...
    '*.png', 'Save Grad-CAM Overlay', ...
    ['GradCAM_' char(traitName) '.png']);

if ~isequal(saveName, 0)

    exportgraphics(gcf, ...
        fullfile(savePath, saveName), ...
        'Resolution', 300);

    fprintf('\nGrad-CAM image saved successfully.\n');

end

save('GradCAM_Results.mat', ...
    'camMap', ...
    'predictedValue', ...
    'traitName', ...
    'targetLayer', ...
    'imagePath');

fprintf('\nGrad-CAM analysis completed.\n');


%% =========================================================
% FUNCTION: TARGET-SPECIFIC GRAD-CAM
% =========================================================

function [camMap, predictedValue] = gradCAM_MTAER( ...
    dlnet, dlImg, targetLayer, outputLayer, ...
    outputIndex, minVals, maxVals)

    % -----------------------------------------------------
    % 1. Get feature maps and selected output in SAME pass
    % -----------------------------------------------------

    [features, score] = forward(dlnet, dlImg, ...
        'Outputs', {char(targetLayer), char(outputLayer)});

    % -----------------------------------------------------
    % 2. Calculate gradient of selected output
    %    with respect to selected feature maps
    % -----------------------------------------------------

    gradients = dlgradient(sum(score, 'all'), features);

    % -----------------------------------------------------
    % 3. Global average pooling of gradients
    % -----------------------------------------------------

    pooledGrad = mean(gradients, [1 2]);

    % -----------------------------------------------------
    % 4. Weighted combination of feature maps
    % -----------------------------------------------------

    cam = sum(features .* pooledGrad, 3);

    % Retain positive contributions
    cam = max(cam, 0);

    % -----------------------------------------------------
    % 5. Convert prediction to original measurement units
    % -----------------------------------------------------

    traitNames = ["Anthocyanin", "TPC", "TFC", "DPPH"];

    minVal = minVals{1, "min_" + traitNames(outputIndex)};

    maxVal = maxVals{1, "max_" + traitNames(outputIndex)};

    rawScore = double(gather(extractdata(score)));

    predictedValue = rawScore .* (maxVal - minVal) + minVal;

    % -----------------------------------------------------
    % 6. Extract and resize activation map
    % -----------------------------------------------------

    camMap = double(gather(extractdata(cam)));

    camMap = squeeze(camMap);

    camMap = imresize(camMap, ...
        dlnet.Layers(1).InputSize(1:2));

    % -----------------------------------------------------
    % 7. Normalize between 0 and 1
    % -----------------------------------------------------

    camMin = min(camMap(:));
    camMax = max(camMap(:));

    if camMax > camMin
        camMap = (camMap - camMin) ./ (camMax - camMin);
    else
        warning('Grad-CAM activation map has no variation.');
        camMap = zeros(size(camMap));
    end

end
