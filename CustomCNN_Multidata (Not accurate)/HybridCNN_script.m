% Final Enhanced CNN for Delonix regia Pigment Prediction (Fully Working)
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

% Normalize
targetNames = ["Anthocyanin", "TPC", "TFC", "DPPH"];
minVals = varfun(@min, labels, 'InputVariables', targetNames);
maxVals = varfun(@max, labels, 'InputVariables', targetNames);

for i = 1:numel(targetNames)
    var = targetNames(i);
    labels.(var + "_norm") = (labels.(var) - minVals.("min_" + var)) ./ ...
                              (maxVals.("max_" + var) - minVals.("min_" + var));
end

labelMatrix = [labels.Anthocyanin_norm, labels.TPC_norm, labels.TFC_norm, labels.DPPH_norm];

%% Step 2: Split Data
rng(1);
labels = labels(randperm(height(labels)), :);
nTrain = round(0.8 * height(labels));
trainData = labels(1:nTrain, :);
valData = labels(nTrain+1:end, :);

%% Step 3: Prepare Datastores
inputSize = [128 128];

trainLabels = [trainData.Anthocyanin_norm, trainData.TPC_norm, trainData.TFC_norm, trainData.DPPH_norm];
labelMap = containers.Map();
for i = 1:height(trainData)
    [~, name, ext] = fileparts(trainData.image(i));
    key = string(name + ext);
    labelMap(key) = trainLabels(i,:);
end

trainImds = imageDatastore(trainData.fullpath);
trainImds.ReadFcn = @(f) augmentAndEnhanceImage(f, inputSize);
trainLabels = [trainData.Anthocyanin_norm, trainData.TPC_norm, trainData.TFC_norm, trainData.DPPH_norm];
dsTrain = combine(trainImds, arrayDatastore(trainLabels));

valLabels = [valData.Anthocyanin_norm, valData.TPC_norm, valData.TFC_norm, valData.DPPH_norm];
valImds = imageDatastore(valData.fullpath);
valImds.ReadFcn = @(f) augmentAndEnhanceImage(f, inputSize);
dsVal = combine(valImds, arrayDatastore(valLabels));

%% Step 4: Define CNN
layers = [
    imageInputLayer([128 128 3],'Normalization','none','Name','input')

    convolution2dLayer(5, 16, 'Stride',1, 'Padding','same','Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','pool1')

    convolution2dLayer(3, 32, 'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2,'Stride',2,'Name','pool2')

    convolution2dLayer(3, 64, 'Padding','same','Name','conv3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')
    averagePooling2dLayer(2,'Stride',2,'Name','avgpool')

    fullyConnectedLayer(128,'Name','fc1')
    reluLayer('Name','relu_fc1')
    dropoutLayer(0.3,'Name','drop1')

    fullyConnectedLayer(64,'Name','fc2')
    reluLayer('Name','relu_fc2')
    dropoutLayer(0.3,'Name','drop2')

    fullyConnectedLayer(4,'Name','fc_output')
    regressionLayer('Name','output')
];

%% Step 5: Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 4, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData', dsVal, ...
    'ValidationFrequency', 30, ...
    'ValidationPatience', 10, ...
    'Plots','training-progress', ...
    'Verbose',false);

%% Step 6: Train
[trainedNet, trainInfo] = trainNetwork(dsTrain, layers, options);

%% Step 7: Save Model
save('HybridCNN_Final_Delonix.mat','trainedNet','trainInfo','minVals','maxVals');

%% Step 8: Evaluate
YPredNorm = predict(trainedNet, dsVal);
YPred = YPredNorm .* (maxVals{1,:} - minVals{1,:}) + minVals{1,:};
YTrue = [valData.Anthocyanin, valData.TPC, valData.TFC, valData.DPPH];

rmse = sqrt(mean((YPred - YTrue).^2));
r2 = 1 - sum((YPred - YTrue).^2) ./ sum((YTrue - mean(YTrue)).^2);

fprintf('\n📊 Final Evaluation (Fully Working Hybrid CNN):\n');
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
sgtitle('Actual vs Predicted - Fully Working Hybrid CNN');

%% Helper Functions
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

function label = lookupLabel(info, labelMap)
    [~, name, ext] = fileparts(info.Filename);
    key = string(name + ext);
    if ~isKey(labelMap, key)
        error("Label not found for " + key);
    end
    label = labelMap(key);
end
