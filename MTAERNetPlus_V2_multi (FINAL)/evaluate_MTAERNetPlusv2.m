clc; clear;

%% Load the model and metadata
load('MTAERNetPlusv2_Final.mat', 'trainedNet', 'minVals', 'maxVals');

%% Load labels and restore validation split
labels = readtable('labels.csv', 'VariableNamingRule', 'preserve');
labels.image = strtrim(string(labels.image));
labels.fullpath = fullfile(pwd, 'images', labels.image);
labels = labels(isfile(labels.fullpath), :);

labels.Anthocyanin = labels.("Anthocyanin (mg/100g)");
labels.TPC         = labels.("TPC (mg GAE/g)");
labels.TFC         = labels.("TFC (mg QE/g)");
labels.DPPH        = labels.("DPPH % Inhibition");

targetNames = ["Anthocyanin", "TPC", "TFC", "DPPH"];
origCols    = ["Anthocyanin (mg/100g)", "TPC (mg GAE/g)", ...
               "TFC (mg QE/g)", "DPPH % Inhibition"];

% Validation split (same as training)
rng(1);
labels = labels(randperm(height(labels)), :);
nTrain = round(0.8 * height(labels));
valData = labels(nTrain+1:end, :);

%% Create validation datastore
inputSize = [224 224 3];
readFcn = @(x) preprocessImage(x, inputSize);
valImds = imageDatastore(valData.fullpath, 'ReadFcn', readFcn);

%% Predict
YPredNorm = predict(trainedNet, valImds);   % 4×N
YPredNorm = squeeze(YPredNorm)';            % N×4

%% Rescale to original units
YPred = zeros(size(YPredNorm));
for i = 1:4
    ymin = minVals{1, i};  % ✅ access by index
    ymax = maxVals{1, i};
    YPred(:,i) = YPredNorm(:,i) * (ymax - ymin) + ymin;
end

YTrue = valData{:, origCols};

%% Calculate RMSE and R²
rmse = sqrt(mean((YPred - YTrue).^2));
r2 = 1 - sum((YPred - YTrue).^2) ./ sum((YTrue - mean(YTrue)).^2);

%% Print Results
fprintf('\n📈 MTAERNetPlusv2 Final Evaluation:\n');
for i = 1:4
    fprintf('%s - RMSE: %.2f, R²: %.4f\n', targetNames(i), rmse(i), r2(i));
end

%% PLOTS: Scatter and Residuals
figure('Name','Prediction Scatter')
for i = 1:4
    subplot(2,2,i)
    scatter(YTrue(:,i), YPred(:,i), 30, 'filled')
    hold on
    plot([min(YTrue(:,i)) max(YTrue(:,i))], ...
         [min(YTrue(:,i)) max(YTrue(:,i))], 'r--', 'LineWidth', 1)
    xlabel('True'); ylabel('Predicted');
    title(sprintf('%s', targetNames(i)))
    grid on; axis equal
end

figure('Name','Prediction Residuals')
for i = 1:4
    subplot(2,2,i)
    res = YPred(:,i) - YTrue(:,i);
    scatter(YPred(:,i), res, 25, 'filled')
    yline(0,'--'); grid on
    xlabel('Predicted'); ylabel('Residual')
    title(sprintf('Residuals: %s', targetNames(i)))
end

%% Helper Function
function out = preprocessImage(file, sz)
    img = imread(file);
    img = im2double(imresize(img, sz(1:2)));
    for i = 1:3
        img(:,:,i) = adapthisteq(img(:,:,i));
    end
    out = im2single(img);
end
