
% RESNET18 SINGLE-OUTPUT REGRESSION
% Anthocyanin prediction in Delonix regia flowers
% MATLAB R2023b

clc;
clear;
close all;

%% STEP 1: LOAD COMMON DATA SPLIT

load('Common_Data_Split.mat', ...
    'trainData', 'valData');

fprintf('Training images: %d\n', height(trainData));
fprintf('Validation images: %d\n', height(valData));

% Create image datastores
imdsTrain = imageDatastore(trainData.fullpath);
imdsVal = imageDatastore(valData.fullpath);


%% STEP 4: NORMALIZE ANTHOCYANIN

% Calculate normalization from training data only
mu = mean(trainData.anthocyanin);
sigma = std(trainData.anthocyanin);

if sigma == 0 || ~isfinite(sigma)
    error('Invalid standard deviation.');
end

trainY = (trainData.anthocyanin - mu) ./ sigma;
valY = (valData.anthocyanin - mu) ./ sigma;

% Actual validation values
labelsVal = valData;


%% STEP 5: IMAGE PREPROCESSING

inputSize = [224 224];

% Random augmentation for training images only
imdsTrain.ReadFcn = @(f) preprocessImage(f, inputSize, true);

% Fixed preprocessing for validation images
imdsVal.ReadFcn = @(f) preprocessImage(f, inputSize, false);

%% STEP 6: CREATE DATASTORES

% Convert response values into array datastores
adsTrain = arrayDatastore(trainY, 'OutputType', 'same');
adsVal = arrayDatastore(valY, 'OutputType', 'same');

% Combine images and corresponding anthocyanin values
cdsTrain = combine(imdsTrain, adsTrain);
cdsVal = combine(imdsVal, adsVal);

%% STEP 7: LOAD PRETRAINED RESNET18

net = resnet18;

lgraph = layerGraph(net);

% Remove original classification layers
lgraph = removeLayers(lgraph, ...
    {'fc1000', 'prob', 'ClassificationLayer_predictions'});

% Add regression output layers
newLayers = [
    fullyConnectedLayer(1, 'Name', 'fcRegression')
    regressionLayer('Name', 'output')
    ];

lgraph = addLayers(lgraph, newLayers);

% Connect regression layer to ResNet18 feature extraction layer
lgraph = connectLayers(lgraph, 'pool5', 'fcRegression');

%% STEP 8: TRAINING OPTIONS

options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 8, ...
    'InitialLearnRate', 1e-5, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', cdsVal, ...
    'ValidationFrequency', 30, ...
    'ValidationPatience', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

%% STEP 9: TRAIN NETWORK

fprintf('\nStarting ResNet18 training...\n');

[trainedNet, trainInfo] = trainNetwork( ...
    cdsTrain, lgraph, options);

fprintf('\nTraining completed successfully.\n');

%% STEP 10: PREDICTION ON VALIDATION DATA

% Predict normalized anthocyanin values
YPredNorm = predict(trainedNet, imdsVal, ...
    'MiniBatchSize', 8);

% Convert predictions to original units
YPred = YPredNorm(:) .* sigma + mu;

% Actual anthocyanin values
YTrue = labelsVal.anthocyanin(:);

%% STEP 11: CALCULATE PERFORMANCE METRICS

% Root Mean Square Error
RMSE = sqrt(mean((YTrue - YPred).^2));

% Mean Absolute Error
MAE = mean(abs(YTrue - YPred));

% Coefficient of determination
SSres = sum((YTrue - YPred).^2);
SStot = sum((YTrue - mean(YTrue)).^2);

if SStot == 0
    R2 = NaN;
else
    R2 = 1 - (SSres / SStot);
end

% Mean Absolute Percentage Error
nonZeroIdx = YTrue ~= 0;

if any(nonZeroIdx)
    MAPE = mean(abs((YTrue(nonZeroIdx) - ...
        YPred(nonZeroIdx)) ./ YTrue(nonZeroIdx))) * 100;
else
    MAPE = NaN;
end

%% STEP 12: DISPLAY RESULTS

fprintf('\n====================================\n');
fprintf('RESNET18 VALIDATION PERFORMANCE\n');
fprintf('====================================\n');

fprintf('RMSE : %.4f mg/100g\n', RMSE);
fprintf('MAE  : %.4f mg/100g\n', MAE);
fprintf('R2   : %.4f\n', R2);
fprintf('MAPE : %.2f %%\n', MAPE);

%% STEP 13: ACTUAL VS PREDICTED PLOT

figure;

scatter(YTrue, YPred, 45, 'filled');
hold on;

minValue = min([YTrue; YPred]);
maxValue = max([YTrue; YPred]);

plot([minValue maxValue], ...
     [minValue maxValue], 'r--', 'LineWidth', 1.5);

xlabel('Actual Anthocyanin (mg/100g)');
ylabel('Predicted Anthocyanin (mg/100g)');

title('ResNet18: Actual vs Predicted Anthocyanin');

legend('Predictions', 'Ideal prediction', ...
    'Location', 'best');

grid on;
axis square;


%% STEP 14: SAVE RESULTS

results = table(YTrue, YPred, ...
    'VariableNames', {'Actual_Anthocyanin', ...
                      'Predicted_Anthocyanin'});

metrics = table(RMSE, MAE, R2, MAPE);

save('ResNet18_Anthocyanin_Final.mat', ...
    'trainedNet', 'trainInfo', 'mu', 'sigma', ...
    'metrics', 'results', 'trainData', 'valData');

writetable(results, ...
    'ResNet18_Validation_Predictions.xlsx');

writetable(metrics, ...
    'ResNet18_Performance_Metrics.xlsx');

fprintf('\nModel and results saved successfully.\n');


%% STEP 15: IMAGE PREPROCESSING FUNCTION

function img = preprocessImage(file, targetSize, augment)

    img = imread(file);

    % Convert grayscale images to RGB
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);
    end

    % Remove extra channels if present
    if size(img, 3) > 3
        img = img(:, :, 1:3);
    end

    % Resize and convert to single precision
    img = im2single(imresize(img, targetSize));

    % Apply augmentation only to training images
    if augment

        % Random rotation
        if rand > 0.5
            angle = -15 + 30 * rand;
            img = imrotate(img, angle, 'bilinear', 'crop');
        end

        % Random horizontal translation
        if rand > 0.5
            dx = randi([-10 10]);
            dy = randi([-10 10]);

            img = imtranslate(img, [dx dy], ...
                'OutputView', 'same');
        end

        % Random horizontal flipping
        if rand > 0.5
            img = fliplr(img);
        end

    end

end
