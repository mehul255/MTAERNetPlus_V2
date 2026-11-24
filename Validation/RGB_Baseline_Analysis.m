
clc;
clear;
close all;

%% 1. Load dataset
labels = readtable("labels.csv", ...
    "VariableNamingRule","preserve");

labels.image = strtrim(string(labels.image));

labels.Anthocyanin = labels.("Anthocyanin (mg/100g)");
labels.TPC = labels.("TPC (mg GAE/g)");
labels.TFC = labels.("TFC (mg QE/g)");
labels.DPPH = labels.("DPPH % Inhibition");

%% 2. Reconstruct the exact original split
rng(1);

labels = labels(randperm(height(labels)),:);

N = height(labels);
nTrain = round(0.70*N);
nVal = round(0.15*N);

trainData = labels(1:nTrain,:);
testData = labels(nTrain+nVal+1:end,:);

fprintf("Training images: %d\n",height(trainData));
fprintf("Test images: %d\n",height(testData));

%% 3. Extract RGB features
imageFolder = "images";

XTrain = zeros(height(trainData),10);
XTest = zeros(height(testData),10);

for i = 1:height(trainData)
    file = fullfile(imageFolder,trainData.image(i));
    XTrain(i,:) = extractRGBFeatures(file);
end

for i = 1:height(testData)
    file = fullfile(imageFolder,testData.image(i));
    XTest(i,:) = extractRGBFeatures(file);
end

fprintf("Training feature matrix: %d x %d\n", ...
    size(XTrain,1),size(XTrain,2));

fprintf("Test feature matrix: %d x %d\n", ...
    size(XTest,1),size(XTest,2));

%% 4. Standardize using training data only
mu = mean(XTrain,1);
sigma = std(XTrain,0,1);
sigma(sigma == 0) = 1;

XTrainStd = (XTrain-mu)./sigma;
XTestStd = (XTest-mu)./sigma;

%% 5. Train RGB ridge regression models
traits = ["Anthocyanin","TPC","TFC","DPPH"];

RMSE = zeros(4,1);
MAE = zeros(4,1);
R2 = zeros(4,1);

RGBPredictions = table(testData.image, ...
    'VariableNames',{'Image'});

for k = 1:numel(traits)

    trait = traits(k);

    YTrain = trainData.(trait);
    YActual = testData.(trait);

    mdl = fitrlinear(XTrainStd,YTrain, ...
        'Learner','leastsquares', ...
        'Regularization','ridge', ...
        'Lambda',0.001, ...
        'Solver','lbfgs');

    YPred = predict(mdl,XTestStd);

    RMSE(k) = sqrt(mean((YPred-YActual).^2));
    MAE(k) = mean(abs(YPred-YActual));

    SSres = sum((YActual-YPred).^2);
    SStot = sum((YActual-mean(YActual)).^2);

    R2(k) = 1-SSres/SStot;

    RGBPredictions.(trait+"_Actual") = YActual;
    RGBPredictions.(trait+"_Predicted") = YPred;

end

%% 6. Display results
RGBResults = table(traits',RMSE,MAE,R2, ...
    'VariableNames',{'Trait','RMSE','MAE','R2'});

disp(RGBResults);

%% 7. Save results
writetable(RGBResults,"RGB_Baseline_Results.csv");
writetable(RGBPredictions,"RGB_Baseline_Predictions.csv");

save("RGB_Baseline_Analysis.mat", ...
    "RGBResults","RGBPredictions");

fprintf("\nRGB baseline analysis completed.\n");
