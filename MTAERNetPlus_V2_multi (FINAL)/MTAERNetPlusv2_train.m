clc; clear; close all;

%% Step 1: Load Labels
labels = readtable('labels.csv', 'VariableNamingRule','preserve');
labels.image = strtrim(string(labels.image));
labels.fullpath = fullfile(pwd, 'images', labels.image);

% Rename long columns
labels.Anthocyanin = labels.("Anthocyanin (mg/100g)");
labels.TPC         = labels.("TPC (mg GAE/g)");
labels.TFC         = labels.("TFC (mg QE/g)");
labels.DPPH        = labels.("DPPH % Inhibition");

% Normalize labels
targetNames = ["Anthocyanin", "TPC", "TFC", "DPPH"];
for i = 1:numel(targetNames)
    labels.(targetNames(i) + "_norm") = rescale(labels.(targetNames(i)), 0, 1);
end

minValsRaw = varfun(@min, labels, 'InputVariables', targetNames);
maxValsRaw = varfun(@max, labels, 'InputVariables', targetNames);
% Rename columns: 'min_Anthocyanin' → 'Anthocyanin'
minVals = renamevars(minValsRaw, minValsRaw.Properties.VariableNames, targetNames);
maxVals = renamevars(maxValsRaw, maxValsRaw.Properties.VariableNames, targetNames);

%% Step 2: Split Train-Val
rng(1);
labels = labels(randperm(height(labels)), :);
nTrain = round(0.8 * height(labels));
trainData = labels(1:nTrain, :);
valData   = labels(nTrain+1:end, :);

%% Step 3: Datastores
inputSize = [224 224 3];
readFcn = @(x) preprocessImage(x, inputSize);
trainLabels = [trainData.Anthocyanin_norm, trainData.TPC_norm, ...
               trainData.TFC_norm, trainData.DPPH_norm]';
valLabels   = [valData.Anthocyanin_norm, valData.TPC_norm, ...
               valData.TFC_norm, valData.DPPH_norm]';

trainImds = imageDatastore(trainData.fullpath, 'ReadFcn', readFcn);
valImds   = imageDatastore(valData.fullpath, 'ReadFcn', readFcn);
dsTrain = combine(trainImds, arrayDatastore(trainLabels, 'IterationDimension', 2));
dsVal   = combine(valImds, arrayDatastore(valLabels, 'IterationDimension', 2));

%% Step 4: Define Novel Architecture (ResNet18 + CBAM + Multi-output)
net = resnet18;
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'});

% CBAM
% CBAM Block
cbam = [
    globalAveragePooling2dLayer('Name','gap')
    fullyConnectedLayer(512,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(512,'Name','fc2')
    sigmoidLayer('Name','sigmoid_cbam')
    multiplicationLayer(2,'Name','cbam_mult')
];
lgraph = addLayers(lgraph, cbam);
lgraph = connectLayers(lgraph,'pool5','gap');
lgraph = connectLayers(lgraph,'sigmoid_cbam','cbam_mult/in2');

% Shared Trunk
shared = [
    fullyConnectedLayer(256,'Name','shared_fc')
    reluLayer('Name','shared_relu')
];
lgraph = addLayers(lgraph, shared);
lgraph = connectLayers(lgraph,'cbam_mult','shared_fc');

% Multi-task Branches
for i = 1:4
    branch = [
        fullyConnectedLayer(64,'Name',sprintf('fc_b%d',i))
        reluLayer('Name',sprintf('relu_b%d',i))
        dropoutLayer(0.3,'Name',sprintf('drop_b%d',i))
        fullyConnectedLayer(1,'Name',sprintf('out_b%d',i))
    ];
    lgraph = addLayers(lgraph, branch);
    lgraph = connectLayers(lgraph,'shared_relu',sprintf('fc_b%d',i));
end

% Concatenation + Regression
concat = concatenationLayer(1, 4, 'Name','concat');
regress = regressionLayer('Name','regression_output');
lgraph = addLayers(lgraph, [concat regress]);
for i = 1:4
    lgraph = connectLayers(lgraph,sprintf('out_b%d',i),sprintf('concat/in%d',i));
end

%% Step 5: Train
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 8, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData', dsVal, ...
    'ValidationPatience', 10, ...
    'Verbose', false, ...
    'Plots','training-progress');

[trainedNet, trainInfo] = trainNetwork(dsTrain, lgraph, options);

%% Step 6: Convert to dlnetwork (for Grad-CAM)
lgraphNoOut = removeLayers(lgraph, 'regression_output');
dlnet = dlnetwork(lgraphNoOut);

%% Step 7: Save All
save('MTAERNetPlusv2_Final.mat','trainedNet','dlnet','trainInfo','lgraph','minVals','maxVals');
