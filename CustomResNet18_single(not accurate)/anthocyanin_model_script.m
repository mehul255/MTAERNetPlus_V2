% Anthocyanin Prediction Training Script using ResNet18
% ------------------------------------------------------
% Requirements:
% - MATLAB Deep Learning Toolbox
% - Image folder: 'images/' containing training images
% - CSV: 'labels.csv' with columns 'image', 'anthocyanin'

%% Step 1: Load Labels
labels = readtable('labels.csv');
labels.image = string(labels.image);

%% Step 2: Normalize Anthocyanin Labels
mu = mean(labels.anthocyanin);
sigma = std(labels.anthocyanin);
labels.anthocyanin_norm = (labels.anthocyanin - mu) / sigma;

%% Step 3: Create Image Datastore
imds = imageDatastore('images', 'FileExtensions', {'.jpg','.png','.jpeg'}, ...
    'IncludeSubfolders', false);
[~, imNames, ext] = cellfun(@fileparts, imds.Files, 'UniformOutput', false);
imNamesFull = strcat(imNames, ext);

%% Step 4: Match Labels with Images
[~, idx] = ismember(imNamesFull, labels.image);
validIdx = idx > 0;
imds.Files = imds.Files(validIdx);
labels = labels(idx(validIdx), :);

%% Step 5: Split Data
numImages = numel(imds.Files);
[trainIdx, valIdx] = dividerand(numImages, 0.8, 0.2);
imdsTrain = subset(imds, trainIdx);
imdsVal = subset(imds, valIdx);
labelsTrain = labels(trainIdx, :);
labelsVal = labels(valIdx, :);

% Step 6: Create Datastores for Regression
trainTbl = table(labelsTrain.anthocyanin_norm, 'VariableNames', {'response'});
valTbl = table(labelsVal.anthocyanin_norm, 'VariableNames', {'response'});

adsTrain = arrayDatastore(table2array(trainTbl));
adsVal = arrayDatastore(table2array(valTbl));

cdsTrain = combine(imdsTrain, adsTrain);
cdsVal = combine(imdsVal, adsVal);


% Step 7: Data Augmentation
augmenter = imageDataAugmenter( ...
    'RandRotation', [-15 15], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10], ...
    'RandXScale', [0.9 1.1], ...
    'RandYScale', [0.9 1.1]);

augImdsTrain = augmentedImageDatastore([224 224], imdsTrain, 'DataAugmentation', augmenter);
augImdsVal = augmentedImageDatastore([224 224], imdsVal);

cdsTrain = combine(augImdsTrain, adsTrain);
cdsVal = combine(augImdsVal, adsVal);


%% Step 8: Load Pretrained Network
net = resnet18;
layers = layerGraph(net);
layers = removeLayers(layers, {'fc1000','ClassificationLayer_predictions'});
newLayers = [
    fullyConnectedLayer(1, 'Name', 'fcRegression')
    regressionLayer('Name', 'output')];
layers = addLayers(layers, newLayers);
layers = connectLayers(layers, 'pool5', 'fcRegression');

%% Step 9: Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 8, ...
    'InitialLearnRate', 1e-5, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', cdsVal, ...
    'ValidationFrequency', 30, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

%% Step 10: Train the Network
trainedNet = trainNetwork(cdsTrain, layers, options);

%% Step 11: Save Model + Normalization Info
save('AnthocyaninPredictor.mat', 'trainedNet', 'mu', 'sigma');

fprintf('✅ Trained and saved as AnthocyaninPredictor.mat\n');
