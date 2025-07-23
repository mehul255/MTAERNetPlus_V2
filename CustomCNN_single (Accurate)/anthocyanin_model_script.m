% Anthocyanin Prediction using Custom CNN
% ----------------------------------------

%% Step 1: Load and Normalize Labels
labels = readtable('labels.csv');
labels.image = string(labels.image);
imageFolder = fullfile(pwd, 'images');
labels.fullpath = fullfile(imageFolder, labels.image);
labels = labels(isfile(labels.fullpath), :);

minLabel = min(labels.anthocyanin);
maxLabel = max(labels.anthocyanin);
labels.labelNorm = (labels.anthocyanin - minLabel) / (maxLabel - minLabel);

%% Step 2: Shuffle and Split Data
rng(1);
labels = labels(randperm(height(labels)), :);
nTrain = round(0.8 * height(labels));
trainData = labels(1:nTrain, :);
valData   = labels(nTrain+1:end, :);

inputSize = [128 128];

%% Step 3: Datastores
trainDs = fileDatastore(trainData.fullpath, 'ReadFcn', @(f) im2single(imresize(imread(f), inputSize)));
valDs   = fileDatastore(valData.fullpath, 'ReadFcn', @(f) im2single(imresize(imread(f), inputSize)));

dsTrain = combine(trainDs, arrayDatastore(trainData.labelNorm));
dsVal   = combine(valDs, arrayDatastore(valData.labelNorm));

%% Step 4: Define Custom CNN Architecture
layers = [
    imageInputLayer([128 128 3],'Normalization','none')

    convolution2dLayer(3, 16, 'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3, 32, 'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3, 64, 'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];

%% Step 5: Output Function to Print Training Info
epochCounter = 0;  % shared via nested function
trainingStartTime = tic;
function stop = customOutputFcn(info)
    stop = false;
    persistent headerPrinted startTime

    if info.State == "start"
        headerPrinted = false;
        startTime = tic;  % Start timer at beginning
    end

    if info.State == "iteration"
        if ~headerPrinted
            fprintf('Epoch  | Iteration | Time Elapsed | Mini-batch RMSE | Val RMSE | Learning Rate\n');
            fprintf('----------------------------------------------------------------------\n');
            headerPrinted = true;
        end

        if ~mod(info.Iteration, 10)  % Print every 10 iterations
            elapsed = toc(startTime);
            fprintf('%6d | %9d | %12s | %15.4f | %8.4f | %.2e\n', ...
                info.Epoch, info.Iteration, ...
                duration(0,0,round(elapsed), "Format", "hh:mm:ss"), ...
                sqrt(info.TrainingLoss), ...
                sqrt(info.ValidationLoss), ...
                info.BaseLearnRate);
        end
    elseif info.State == "done"
        fprintf('Training completed.\n');
    end
end

%% Step 6: Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', dsVal, ...
    'ValidationFrequency', 20, ...
    'ValidationPatience', 5, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'OutputFcn', @customOutputFcn, ...
    'CheckpointPath', fullfile(pwd, 'checkpoints'));

%% Step 7: Train Model
trainingStartTime = tic; 
[trainedNet, trainInfo] = trainNetwork(dsTrain, layers, options);

%% Step 8: Save Model
save('CustomCNN_AnthocyaninNet.mat', 'trainedNet', 'trainInfo', 'minLabel', 'maxLabel');

%% Step 9: Evaluate Model
YPredNorm = predict(trainedNet, dsVal);
YPred = YPredNorm * (maxLabel - minLabel) + minLabel;
YTrue = valData.anthocyanin;

rmse = sqrt(mean((YPred - YTrue).^2));
r2 = 1 - sum((YTrue - YPred).^2) / sum((YTrue - mean(YTrue)).^2);

fprintf('\n📊 Final Evaluation:\n');
fprintf('RMSE: %.2f mg\n', rmse);
fprintf('R²: %.4f\n', r2);

figure;
scatter(YTrue, YPred, 50, 'filled');
refline(1, 0);
xlabel('Actual Anthocyanin (mg/100g)');
ylabel('Predicted');
title('Actual vs Predicted - Custom CNN');
grid on;
