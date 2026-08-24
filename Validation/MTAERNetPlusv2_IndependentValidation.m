clc;
clear;
close all;

% MTAERNetPlusv2
% independent test-set validation
%
% Dataset:
%   70% Training
%   15% Validation
%   15% Independent Test
%
% Outputs:
%   Anthocyanin
%   TPC
%   TFC
%   DPPH

%% Step 1: Load Labels
labels = readtable('labels.csv', ...
    'VariableNamingRule','preserve');

labels.image = strtrim(string(labels.image));

% Image paths
labels.fullpath = fullfile(pwd, 'images', labels.image);

%% Check that images exist
fprintf('\nChecking image files...\n');

missingImages = ~isfile(labels.fullpath);

if any(missingImages)
    fprintf('WARNING: %d image files are missing.\n', ...
        sum(missingImages));

    disp(labels.image(missingImages));

    error('Missing image files detected. Check the images folder.');
else
    fprintf('All %d image files found.\n', height(labels));
end


%% Step 2: Rename target variables

labels.Anthocyanin = labels.("Anthocyanin (mg/100g)");
labels.TPC         = labels.("TPC (mg GAE/g)");
labels.TFC         = labels.("TFC (mg QE/g)");
labels.DPPH        = labels.("DPPH % Inhibition");

targetNames = ["Anthocyanin", ...
               "TPC", ...
               "TFC", ...
               "DPPH"];

%% Check missing values

for i = 1:numel(targetNames)

    target = labels.(targetNames(i));

    if any(isnan(target))
        error('Missing values detected in %s.', ...
            targetNames(i));
    end

end

fprintf('No missing target values detected.\n');


%% ============================================================
% Step 3: Reproducible Dataset Split
% =============================================================

rng(1);

% Randomize observations
labels = labels(randperm(height(labels)), :);

N = height(labels);

fprintf('\nTotal observations: %d\n', N);

% 70% training
% 15% validation
% 15% independent test

nTrain = round(0.70 * N);
nVal   = round(0.15 * N);

trainData = labels(1:nTrain, :);

valStart = nTrain + 1;
valEnd   = nTrain + nVal;

valData = labels(valStart:valEnd, :);

testData = labels(valEnd+1:end, :);

fprintf('Training samples   : %d\n', height(trainData));
fprintf('Validation samples : %d\n', height(valData));
fprintf('Test samples       : %d\n', height(testData));


%% ============================================================
% Step 4: Calculate NORMALIZATION PARAMETERS
% ONLY FROM TRAINING DATA
% =============================================================

minVals = zeros(1, numel(targetNames));
maxVals = zeros(1, numel(targetNames));

for i = 1:numel(targetNames)

    target = trainData.(targetNames(i));

    minVals(i) = min(target);
    maxVals(i) = max(target);

end

fprintf('\nTraining-set normalization parameters:\n');

for i = 1:numel(targetNames)

    fprintf('%s: Min = %.6f, Max = %.6f\n', ...
        targetNames(i), ...
        minVals(i), ...
        maxVals(i));

end


%% ============================================================
% Step 5: Normalize Targets
% USING TRAINING MIN/MAX
% =============================================================

trainTargetsRaw = zeros(height(trainData),4);
valTargetsRaw   = zeros(height(valData),4);
testTargetsRaw  = zeros(height(testData),4);

for i = 1:numel(targetNames)

    trainTargetsRaw(:,i) = trainData.(targetNames(i));
    valTargetsRaw(:,i)   = valData.(targetNames(i));
    testTargetsRaw(:,i)  = testData.(targetNames(i));

end


% Avoid division by zero
rangeVals = maxVals - minVals;

if any(rangeVals == 0)
    error('One or more target variables have zero range.');
end


% Min-max normalization using TRAINING statistics

trainTargetsNorm = ...
    (trainTargetsRaw - minVals) ./ rangeVals;

valTargetsNorm = ...
    (valTargetsRaw - minVals) ./ rangeVals;

testTargetsNorm = ...
    (testTargetsRaw - minVals) ./ rangeVals;


%% ============================================================
% Step 6: Image Preprocessing
% =============================================================

inputSize = [224 224 3];

readFcn = @(x) preprocessImage(x, inputSize);


%% ============================================================
% Step 7: Create Datastores
% =============================================================

trainImds = imageDatastore( ...
    trainData.fullpath, ...
    'ReadFcn', readFcn);

valImds = imageDatastore( ...
    valData.fullpath, ...
    'ReadFcn', readFcn);

testImds = imageDatastore( ...
    testData.fullpath, ...
    'ReadFcn', readFcn);


trainLabels = trainTargetsNorm';

valLabels = valTargetsNorm';

testLabels = testTargetsNorm';


dsTrain = combine( ...
    trainImds, ...
    arrayDatastore( ...
        trainLabels, ...
        'IterationDimension',2));


dsVal = combine( ...
    valImds, ...
    arrayDatastore( ...
        valLabels, ...
        'IterationDimension',2));


dsTest = combine( ...
    testImds, ...
    arrayDatastore( ...
        testLabels, ...
        'IterationDimension',2));


%% ============================================================
% Step 8: Define MTAERNetPlusv2 Architecture
% Original ResNet18 + CBAM + Multi-output architecture
% =============================================================

fprintf('\nBuilding MTAERNetPlusv2 architecture...\n');

net = resnet18;

lgraph = layerGraph(net);


%% Remove original classification layers

lgraph = removeLayers( ...
    lgraph, ...
    {'fc1000', ...
     'prob', ...
     'ClassificationLayer_predictions'});


%% ============================================================
% CBAM / Attention Block
% =============================================================

cbam = [

    globalAveragePooling2dLayer( ...
        'Name','gap')

    fullyConnectedLayer( ...
        512, ...
        'Name','fc1')

    reluLayer( ...
        'Name','relu1')

    fullyConnectedLayer( ...
        512, ...
        'Name','fc2')

    sigmoidLayer( ...
        'Name','sigmoid_cbam')

    multiplicationLayer( ...
        2, ...
        'Name','cbam_mult')

    ];


lgraph = addLayers(lgraph, cbam);


%% Connect attention block

lgraph = connectLayers( ...
    lgraph, ...
    'pool5', ...
    'gap');


lgraph = connectLayers( ...
    lgraph, ...
    'sigmoid_cbam', ...
    'cbam_mult/in2');


%% ============================================================
% Shared Feature Trunk
% =============================================================

shared = [

    fullyConnectedLayer( ...
        256, ...
        'Name','shared_fc')

    reluLayer( ...
        'Name','shared_relu')

    ];


lgraph = addLayers( ...
    lgraph, ...
    shared);


lgraph = connectLayers( ...
    lgraph, ...
    'cbam_mult', ...
    'shared_fc');


%% ============================================================
% Four Independent Regression Branches
% =============================================================

for i = 1:4

    branch = [

        fullyConnectedLayer( ...
            64, ...
            'Name',sprintf('fc_b%d',i))

        reluLayer( ...
            'Name',sprintf('relu_b%d',i))

        dropoutLayer( ...
            0.3, ...
            'Name',sprintf('drop_b%d',i))

        fullyConnectedLayer( ...
            1, ...
            'Name',sprintf('out_b%d',i))

        ];

    lgraph = addLayers( ...
        lgraph, ...
        branch);


    lgraph = connectLayers( ...
        lgraph, ...
        'shared_relu', ...
        sprintf('fc_b%d',i));

end


%% ============================================================
% Concatenation + Regression Layer
% =============================================================

concat = concatenationLayer( ...
    1, ...
    4, ...
    'Name','concat');


regress = regressionLayer( ...
    'Name','regression_output');


lgraph = addLayers( ...
    lgraph, ...
    [concat regress]);


for i = 1:4

    lgraph = connectLayers( ...
        lgraph, ...
        sprintf('out_b%d',i), ...
        sprintf('concat/in%d',i));

end


%% ============================================================
% Architecture visualization
% =============================================================

figure;

plot(lgraph);

title('MTAERNetPlusv2 Architecture');


%% ============================================================
% Step 9: Training Options
% % =============================================================

options = trainingOptions( ...
    'adam', ...
    'MaxEpochs',50, ...
    'MiniBatchSize',8, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',dsVal, ...
    'ValidationPatience',10, ...
    'Verbose',true, ...
    'Plots','training-progress');


%% ============================================================
% Step 10: Train Model
% =============================================================

fprintf('\n====================================================\n');
fprintf('Training MTAERNetPlusv2\n');
fprintf('====================================================\n\n');

[trainedNet, trainInfo] = trainNetwork( ...
    dsTrain, ...
    lgraph, ...
    options);


%% ============================================================
% Step 11: Predict on INDEPENDENT TEST SET
% =============================================================

fprintf('\n====================================================\n');
fprintf('Evaluating independent test set\n');
fprintf('====================================================\n');

reset(dsTest);

YPredNorm = predict( ...
    trainedNet, ...
    dsTest);

YPredNorm = double(YPredNorm);

% MATLAB returns:
% 4 × 1 × 1 × N
%
% Convert to:
% N × 4

YPredNorm = squeeze(YPredNorm);

% After squeeze:
% 4 × N

YPredNorm = YPredNorm';

fprintf('Prediction matrix size: %d × %d\n', ...
    size(YPredNorm,1), ...
    size(YPredNorm,2));


%% ============================================================
% Step 12: Convert Predictions Back to Original Units
% =============================================================

YPred = zeros(size(YPredNorm));

for i = 1:4

    YPred(:,i) = ...
        YPredNorm(:,i) .* rangeVals(i) + minVals(i);

end


YTrue = testTargetsRaw;


%% ============================================================
% Step 13: Calculate Performance Metrics
% =============================================================

RMSE = zeros(4,1);
MAE  = zeros(4,1);
R2   = zeros(4,1);
MAPE = zeros(4,1);
MaxAE = zeros(4,1);


for i = 1:4

    actual = YTrue(:,i);

    predicted = YPred(:,i);


    % Error
    errors = predicted - actual;


    % RMSE
    RMSE(i) = sqrt( ...
        mean(errors.^2));


    % MAE
    MAE(i) = mean( ...
        abs(errors));


    % R2
    SSres = sum( ...
        (actual - predicted).^2);

    SStot = sum( ...
        (actual - mean(actual)).^2);

    R2(i) = 1 - SSres/SStot;


    % MAPE
    nonZero = actual ~= 0;

    MAPE(i) = mean( ...
        abs((actual(nonZero) - ...
        predicted(nonZero)) ./ ...
        actual(nonZero))) * 100;


    % Maximum absolute error
    MaxAE(i) = max( ...
        abs(errors));

end


%% ============================================================
% Step 14: Display Final Test Performance
% =============================================================

Results = table( ...
    targetNames', ...
    RMSE, ...
    MAE, ...
    R2, ...
    MAPE, ...
    MaxAE, ...
    'VariableNames', ...
    {'Trait','RMSE','MAE','R2','MAPE_percent','MaxAE'});


fprintf('\n====================================================\n');
fprintf('INDEPENDENT TEST SET PERFORMANCE\n');
fprintf('====================================================\n\n');

disp(Results);


%% ============================================================
% Step 15: Save Results
% =============================================================

writetable( ...
    Results, ...
    'MTAERNetPlusv2_IndependentTest_Results.csv');


%% ============================================================
% Step 16: Save Predictions
% =============================================================

PredictionTable = table( ...
    testData.image, ...
    YTrue(:,1), ...
    YPred(:,1), ...
    YTrue(:,2), ...
    YPred(:,2), ...
    YTrue(:,3), ...
    YPred(:,3), ...
    YTrue(:,4), ...
    YPred(:,4), ...
    'VariableNames', ...
    {'Image', ...
     'Anthocyanin_Actual', ...
     'Anthocyanin_Predicted', ...
     'TPC_Actual', ...
     'TPC_Predicted', ...
     'TFC_Actual', ...
     'TFC_Predicted', ...
     'DPPH_Actual', ...
     'DPPH_Predicted'});


writetable( ...
    PredictionTable, ...
    'MTAERNetPlusv2_Test_Predictions.csv');


%% ============================================================
% Step 17: Predicted vs Actual Plots
% =============================================================

traitTitles = { ...
    'Anthocyanin (mg/100 g DW)', ...
    'TPC (mg GAE/g)', ...
    'TFC (mg QE/g)', ...
    'DPPH (% inhibition)'};


for i = 1:4

    figure;

    scatter( ...
        YTrue(:,i), ...
        YPred(:,i), ...
        35, ...
        'filled');

    hold on;


    minValue = min( ...
        [YTrue(:,i); YPred(:,i)]);

    maxValue = max( ...
        [YTrue(:,i); YPred(:,i)]);


    plot( ...
        [minValue maxValue], ...
        [minValue maxValue], ...
        'k--', ...
        'LineWidth',1.5);


    xlabel( ...
        ['Measured ' traitTitles{i}]);


    ylabel( ...
        ['Predicted ' traitTitles{i}]);


    title( ...
        ['Independent Test Set: ' traitTitles{i}]);


    grid on;


    text( ...
        0.05, ...
        0.90, ...
        sprintf('R^2 = %.4f\nRMSE = %.4f\nMAE = %.4f', ...
        R2(i),RMSE(i),MAE(i)), ...
        'Units','normalized', ...
        'FontSize',11);


    hold off;

end


%% ============================================================
% Step 18: Residual Plots
% =============================================================

for i = 1:4

    residuals = ...
        YPred(:,i) - YTrue(:,i);


    figure;

    scatter( ...
        YTrue(:,i), ...
        residuals, ...
        35, ...
        'filled');

    hold on;


    yline( ...
        0, ...
        'k--', ...
        'LineWidth',1.5);


    xlabel( ...
        ['Measured ' traitTitles{i}]);


    ylabel('Residual (Predicted - Measured)');


    title( ...
        ['Residual Plot: ' traitTitles{i}]);


    grid on;


    hold off;

end


%% ============================================================
% Step 19: Error Distribution
% =============================================================

for i = 1:4

    errors = ...
        YPred(:,i) - YTrue(:,i);


    figure;

    histogram(errors);


    xlabel('Prediction Error');

    ylabel('Frequency');


    title( ...
        ['Prediction Error Distribution: ' ...
        traitTitles{i}]);


    grid on;

end


%% ============================================================
% Step 20: Save Final Model
% =============================================================

save( ...
    'MTAERNetPlusv2_IndependentValidated.mat', ...
    'trainedNet', ...
    'trainInfo', ...
    'lgraph', ...
    'minVals', ...
    'maxVals', ...
    'rangeVals', ...
    'Results', ...
    'PredictionTable');


fprintf('\n====================================================\n');
fprintf('COMPLETED SUCCESSFULLY\n');
fprintf('====================================================\n');

fprintf('\nTraining samples   : %d\n',height(trainData));
fprintf('Validation samples : %d\n',height(valData));
fprintf('Independent test   : %d\n',height(testData));

fprintf('\nResults saved as:\n');
fprintf('  MTAERNetPlusv2_IndependentTest_Results.csv\n');
fprintf('  MTAERNetPlusv2_Test_Predictions.csv\n');
fprintf('  MTAERNetPlusv2_IndependentValidated.mat\n');


%% ============================================================
% LOCAL FUNCTION
% =============================================================

function I = preprocessImage(filename,inputSize)

    I = imread(filename);

    % Convert grayscale image to RGB
    if size(I,3) == 1
        I = repmat(I,[1 1 3]);
    end

    % Convert RGBA to RGB if required
    if size(I,3) > 3
        I = I(:,:,1:3);
    end

    % Resize
    I = imresize(I,inputSize(1:2));

    % Convert to single precision
    I = im2single(I);

end