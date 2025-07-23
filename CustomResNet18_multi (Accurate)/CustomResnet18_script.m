% ResNet18 + Multi-Output Branch Model with Grad-CAM for Delonix regia Pigments
% ------------------------------------------------------------------------------

clc; clear; close all;

%% Step 1: Load and Normalize Labels
labels = readtable('labels.csv', 'VariableNamingRule', 'preserve');
labels.image = strtrim(string(labels.image));
imageFolder = fullfile(pwd, 'images');
labels.fullpath = fullfile(imageFolder, labels.image);
labels = labels(isfile(labels.fullpath), :);

labels.Anthocyanin = labels.("Anthocyanin (mg/100g)");
labels.TPC = labels.("TPC (mg GAE/g)");
labels.TFC = labels.("TFC (mg QE/g)");
labels.DPPH = labels.("DPPH % Inhibition");

% Normalize to [0,1]
targetNames = ["Anthocyanin", "TPC", "TFC", "DPPH"];

minVals = table();
maxVals = table();
for i = 1:numel(targetNames)
    var = targetNames(i);
    minVals.(var) = min(labels.(var));
    maxVals.(var) = max(labels.(var));
    labels.(var + "_norm") = (labels.(var) - minVals.(var)) ./ (maxVals.(var) - minVals.(var));
end

%% Step 2: Split Data
rng(1);
labels = labels(randperm(height(labels)), :);
nTrain = round(0.8 * height(labels));
trainData = labels(1:nTrain, :);
valData = labels(nTrain+1:end, :);

%% Step 3: Prepare Datastores
inputSize = [224 224];
trainLabels = [trainData.Anthocyanin_norm, trainData.TPC_norm, trainData.TFC_norm, trainData.DPPH_norm];
trainImds = imageDatastore(trainData.fullpath);
trainImds.ReadFcn = @(f) augmentAndEnhanceImage(f, inputSize);
dsTrain = combine(trainImds, arrayDatastore(trainLabels));

valLabels = [valData.Anthocyanin_norm, valData.TPC_norm, valData.TFC_norm, valData.DPPH_norm];
valImds = imageDatastore(valData.fullpath);
valImds.ReadFcn = @(f) augmentAndEnhanceImage(f, inputSize);
dsVal = combine(valImds, arrayDatastore(valLabels));

%% Step 4: Load and Modify ResNet18
net = resnet18;
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'});

branchLayers = [
    fullyConnectedLayer(128,'Name','fc_shared')
    reluLayer('Name','relu_shared')
    fullyConnectedLayer(64,'Name','fc_regress')
    reluLayer('Name','relu_regress')
    dropoutLayer(0.3,'Name','drop')
    fullyConnectedLayer(4,'Name','fc_final')
    regressionLayer('Name','regressionoutput')
];

lgraph = addLayers(lgraph, branchLayers);
lgraph = connectLayers(lgraph,'pool5','fc_shared');

%% Step 5: Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 60, ...
    'MiniBatchSize', 4, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData', dsVal, ...
    'ValidationFrequency', 30, ...
    'ValidationPatience', 10, ...
    'Plots','training-progress', ...
    'Verbose',false);

%% Step 6: Train
[trainedNet, trainInfo] = trainNetwork(dsTrain, lgraph, options);

%% Step 7: Save
save('ResNet18_MultiOutput_Delonix.mat','trainedNet','trainInfo','minVals','maxVals');

%% Step 8: Evaluate
YTrue = [valData.Anthocyanin, valData.TPC, valData.TFC, valData.DPPH];
YPredNorm = predict(trainedNet, dsVal);
YPred = YPredNorm .* (maxVals{1,:} - minVals{1,:}) + minVals{1,:};
YTrue = [valData.Anthocyanin, valData.TPC, valData.TFC, valData.DPPH];

rmse = sqrt(mean((YPred - YTrue).^2));
r2 = 1 - sum((YPred - YTrue).^2) ./ sum((YTrue - mean(YTrue)).^2);

fprintf('\n📊 ResNet18 Final Evaluation:\n');
for i = 1:4
    fprintf('%s - RMSE: %.2f, R^2: %.4f\n', targetNames(i), rmse(i), r2(i));
end

figure;
for i = 1:4
    subplot(2,2,i);
    scatter(YTrue(:,i), YPred(:,i), 40, 'filled');
    refline(1,0);
    title(targetNames(i));
    xlabel('True'); ylabel('Predicted'); grid on;
end
sgtitle('ResNet18 Multi-Output Regression - Gulmohar Pigments');

%% Step 9: Grad-CAM Visualizer for Last Convolution Layer
idx = 1; % Show Grad-CAM for first validation image
exampleImage = valImds.Files{idx};
img = imresize(im2single(imread(exampleImage)), inputSize);

% Use dlnetwork for Grad-CAM
dlnet = dlnetwork(trainedNet);
layerName = 'res5b_relu';

featureMap = activations(dlnet, img, layerName);
weights = mean(featureMap, [1 2]);
cMap = sum(featureMap .* reshape(weights, 1, 1, []), 3);
cMap = rescale(cMap);

% Show Grad-CAM overlay
figure;
imshow(img);
hold on;
imagesc(cMap, 'AlphaData', 0.5);
colormap jet;
colorbar;
title('Grad-CAM (ResNet18, First Validation Image)');

%% Helper Function
function out = augmentAndEnhanceImage(file, targetSize)
    img = imread(file);
    img = im2double(imresize(img, targetSize));
    for i = 1:3
        img(:,:,i) = adapthisteq(img(:,:,i));
    end
    if rand > 0.5
        img = imrotate(img, randi([-10,10]), 'bilinear','crop');
    end
    if rand > 0.5
        factor = 0.85 + 0.3*rand();
        img = min(img * factor, 1.0);
    end
    if rand > 0.5
        img = imtranslate(img, [randi([-5 5]) randi([-5 5])]);
    end
    out = im2single(img);
end
