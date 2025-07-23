labels = readtable('labels.csv', 'VariableNamingRule','preserve');
labels.image = strtrim(string(labels.image));
imageFolder = fullfile(pwd, 'images');
labels.fullpath = fullfile(imageFolder, labels.image);

% Rename columns
labels.Anthocyanin = labels.("Anthocyanin (mg/100g)");
labels.TPC         = labels.("TPC (mg GAE/g)");
labels.TFC         = labels.("TFC (mg QE/g)");
labels.DPPH        = labels.("DPPH % Inhibition");

% Normalize target variables
targetNames = ["Anthocyanin", "TPC", "TFC", "DPPH"];
minVals = table(); maxVals = table();
for i = 1:numel(targetNames)
    v = targetNames(i);
    minVals.(v) = min(labels.(v));
    maxVals.(v) = max(labels.(v));
    labels.(v + "_norm") = (labels.(v) - minVals.(v)) / (maxVals.(v) - minVals.(v));
end

rng(1);
labels = labels(randperm(height(labels)), :);
nTrain = round(0.8 * height(labels));
trainData = labels(1:nTrain, :);
valData   = labels(nTrain+1:end, :);

% Ensure files exist
trainData = trainData(isfile(trainData.fullpath), :);
valData   = valData(isfile(valData.fullpath), :);

inputSize = [224 224];  % Required for MobileNetV2
trainLabels = [trainData.Anthocyanin_norm, trainData.TPC_norm, ...
               trainData.TFC_norm, trainData.DPPH_norm];
valLabels   = [valData.Anthocyanin_norm, valData.TPC_norm, ...
               valData.TFC_norm, valData.DPPH_norm];

% Augmentation
augmentFcn = @(f) augmentImage(f, inputSize);
trainImds = imageDatastore(trainData.fullpath, 'ReadFcn', augmentFcn);
valImds   = imageDatastore(valData.fullpath, 'ReadFcn', @(f) im2single(imresize(imread(f), inputSize)));

dsTrain = combine(trainImds, arrayDatastore(trainLabels));
dsVal   = combine(valImds, arrayDatastore(valLabels));

net = mobilenetv2;
lgraph = layerGraph(net);

% Remove final classification layers
lgraph = removeLayers(lgraph, {'Logits','Logits_softmax','ClassificationLayer_Logits'});

% Add custom regression layers
regressionHead = [
    fullyConnectedLayer(64,'Name','fc_regress1')
    reluLayer('Name','relu1')
    dropoutLayer(0.3,'Name','dropout')
    fullyConnectedLayer(4,'Name','fc_final')  % 4 outputs
    regressionLayer('Name','regressionoutput')
];

% Connect to bottleneck
lgraph = addLayers(lgraph, regressionHead);
lgraph = connectLayers(lgraph, 'global_average_pooling2d_1', 'fc_regress1');

options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 8, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', dsVal, ...
    'ValidationFrequency', 20, ...
    'ValidationPatience', 10, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

[trainedNet, trainInfo] = trainNetwork(dsTrain, lgraph, options);
save('MobileNetV2_Regression_Model.mat', 'trainedNet', 'trainInfo', 'minVals', 'maxVals');

YPredNorm = predict(trainedNet, dsVal);
% Denormalize if needed using your saved min/max values

function out = augmentImage(file, targetSize)
    img = imread(file);
    img = imresize(img, targetSize);
    img = im2double(img);
    for i = 1:3
        img(:,:,i) = adapthisteq(img(:,:,i));
    end
    if rand > 0.5
        img = fliplr(img);
    end
    if rand > 0.5
        img = imtranslate(img, [randi([-5 5]) randi([-5 5])]);
    end
    out = im2single(img);
end
