%% MTAERNetPlusv2_ABLATION_MASTER.m
% Four controlled ablation experiments using one common 70:15:15 split.
% E1: no attention, four single-output models
% E2: channel attention, four single-output models
% E3: no attention, one multi-output model
% E4: channel attention, one multi-output model
%
% Requires labels.csv, an images folder, Deep Learning Toolbox and resnet18.
% The retained module is channel attention, NOT full CBAM.

clc; clear; close all;
rng(1,'twister');

%% 1. Settings
labelFile = 'labels.csv';
imageFolder = fullfile(pwd,'images');
inputSize = [224 224 3];
sourceNames = {'Anthocyanin (mg/100g)','TPC (mg GAE/g)', ...
               'TFC (mg QE/g)','DPPH % Inhibition'};
targetNames = {'Anthocyanin','TPC','TFC','DPPH'};
outputFolder = 'Ablation_Results';
if ~exist(outputFolder,'dir'), mkdir(outputFolder); end

%% 2. Load and validate data
if ~isfile(labelFile), error('Cannot find labels.csv in the current folder.'); end
if ~isfolder(imageFolder), error('Cannot find the images folder.'); end
labels = readtable(labelFile,'VariableNamingRule','preserve');
required = [{'image'},sourceNames];
missing = setdiff(required,labels.Properties.VariableNames);
if ~isempty(missing)
    error('Missing labels.csv column(s): %s',strjoin(missing,', '));
end
labels.image = strtrim(string(labels.image));
labels.fullpath = fullfile(imageFolder,labels.image);
Yraw = zeros(height(labels),4);
for k = 1:4, Yraw(:,k) = labels.(sourceNames{k}); end
valid = all(isfinite(Yraw),2) & isfile(labels.fullpath);
if any(~valid)
    fprintf('Removing %d row(s) with missing targets or images.\n',sum(~valid));
    labels = labels(valid,:); Yraw = Yraw(valid,:);
end
N = height(labels);
if N < 20, error('Too few valid images for the requested data split.'); end
fprintf('Valid images: %d\n',N);

%% 3. Create one common split; reused by all ten trainings
order = randperm(N);
nTrain = floor(0.70*N);
nVal = floor(0.15*N);
trainIdx = order(1:nTrain);
valIdx = order(nTrain+1:nTrain+nVal);
testIdx = order(nTrain+nVal+1:end);
trainData = labels(trainIdx,:);
valData = labels(valIdx,:);
testData = labels(testIdx,:);
YtrainRaw = Yraw(trainIdx,:);
YvalRaw = Yraw(valIdx,:);
YtestRaw = Yraw(testIdx,:);
fprintf('Training: %d | Validation: %d | Independent test: %d\n', ...
    numel(trainIdx),numel(valIdx),numel(testIdx));

%% 4. Normalize targets using training subset only
minVals = min(YtrainRaw,[],1);
maxVals = max(YtrainRaw,[],1);
rangeVals = maxVals-minVals;
if any(rangeVals==0), error('A target has zero range in the training set.'); end
YtrainNorm = (YtrainRaw-minVals)./rangeVals;
YvalNorm = (YvalRaw-minVals)./rangeVals;
YtestNorm = (YtestRaw-minVals)./rangeVals;

% Save the exact split and normalization values for reproducibility.
Set = [repmat("Train",numel(trainIdx),1); ...
       repmat("Validation",numel(valIdx),1); ...
       repmat("Test",numel(testIdx),1)];
Image = [labels.image(trainIdx);labels.image(valIdx);labels.image(testIdx)];
SplitTable = table(Image,Set);
writetable(SplitTable,fullfile(outputFolder,'Common_Data_Split.csv'));
save(fullfile(outputFolder,'Common_Data_Split.mat'), ...
    'trainIdx','valIdx','testIdx','minVals','maxVals','rangeVals', ...
    'targetNames','sourceNames','SplitTable');

%% 5. Identical image preprocessing and datastore setup
readFcn = @(f) preprocessImage(f,inputSize);
trainImds = imageDatastore(trainData.fullpath,'ReadFcn',readFcn);
valImds = imageDatastore(valData.fullpath,'ReadFcn',readFcn);
testImds = imageDatastore(testData.fullpath,'ReadFcn',readFcn);

%% 6. Identical training options for all experiments

experimentNames = {'E1_NoAttention_SingleOutput', ...
    'E2_ChannelAttention_SingleOutput', ...
    'E3_NoAttention_MultiOutput', ...
    'E4_ChannelAttention_MultiOutput'};
useAttentionList = [false true false true];
multiOutputList = [false false true true];
allMetricTables = cell(4,1);

%% 7. Run the four experiments
for expNo = 1:4
    expName = experimentNames{expNo};
    useAttention = useAttentionList(expNo);
    multiOutput = multiOutputList(expNo);
    expFolder = fullfile(outputFolder,expName);
    if ~exist(expFolder,'dir'), mkdir(expFolder); end
    fprintf('\n========== %s ==========\n',expName);

    YpredNorm = zeros(numel(testIdx),4);

    if multiOutput
        dsTrain = combine(trainImds,arrayDatastore(YtrainNorm', ...
            'IterationDimension',2));
        dsVal = combine(valImds,arrayDatastore(YvalNorm', ...
            'IterationDimension',2));
        dsTest = combine(testImds,arrayDatastore(YtestNorm', ...
            'IterationDimension',2));
        reset(trainImds); reset(valImds); reset(testImds);
        lgraph = buildAblationNetwork(useAttention,true);
        opts = makeTrainingOptions(dsVal);
        [trainedNet,trainInfo] = trainNetwork(dsTrain,lgraph,opts);
        reset(dsTest);
        rawPred = predict(trainedNet,dsTest);
        YpredNorm = reshape(double(rawPred),4,[])';
        save(fullfile(expFolder,'Trained_MultiOutput_Model.mat'), ...
            'trainedNet','trainInfo','lgraph','minVals','maxVals','rangeVals', ...
            'targetNames','-v7.3');
    else
        for traitNo = 1:4
            fprintf('\nTraining %s single-output model\n',targetNames{traitNo});
            dsTrain = combine(trainImds,arrayDatastore( ...
                YtrainNorm(:,traitNo)','IterationDimension',2));
            dsVal = combine(valImds,arrayDatastore( ...
                YvalNorm(:,traitNo)','IterationDimension',2));
            dsTest = combine(testImds,arrayDatastore( ...
                YtestNorm(:,traitNo)','IterationDimension',2));
            reset(trainImds); reset(valImds); reset(testImds);
            lgraph = buildAblationNetwork(useAttention,false);
            opts = makeTrainingOptions(dsVal);
            [trainedNet,trainInfo] = trainNetwork(dsTrain,lgraph,opts);
            reset(dsTest);
            rawPred = predict(trainedNet,dsTest);
            YpredNorm(:,traitNo) = reshape(double(rawPred),[],1);
            save(fullfile(expFolder,['Model_' targetNames{traitNo} '.mat']), ...
                'trainedNet','trainInfo','lgraph','minVals','maxVals', ...
                'rangeVals','targetNames','-v7.3');
            clear trainedNet trainInfo lgraph rawPred dsTrain dsVal dsTest opts
        end
    end

    %% 8. Independent test metrics in original measurement units
    Ypred = YpredNorm.*rangeVals+minVals;
    Ytrue = YtestRaw;
    RMSE = zeros(4,1); MAE = zeros(4,1); R2 = zeros(4,1);
    MAPE = nan(4,1); MaxAE = zeros(4,1);
    for t = 1:4
        actual = Ytrue(:,t); predicted = Ypred(:,t);
        err = predicted-actual;
        RMSE(t) = sqrt(mean(err.^2));
        MAE(t) = mean(abs(err));
        ssRes = sum((actual-predicted).^2);
        ssTot = sum((actual-mean(actual)).^2);
        if ssTot>0, R2(t)=1-ssRes/ssTot; else, R2(t)=NaN; end
        nz = actual~=0;
        if any(nz), MAPE(t)=mean(abs(err(nz)./actual(nz)))*100; end
        MaxAE(t)=max(abs(err));
    end
    Model = repmat(string(expName),4,1);
    Trait = string(targetNames(:));
    Results = table(Model,Trait,RMSE,MAE,R2,MAPE,MaxAE);
    writetable(Results,fullfile(expFolder,'Independent_Test_Metrics.csv'));

    PredictionTable = table(testData.image, ...
        Ytrue(:,1),Ypred(:,1),Ytrue(:,2),Ypred(:,2), ...
        Ytrue(:,3),Ypred(:,3),Ytrue(:,4),Ypred(:,4), ...
        'VariableNames',{'Image','Anthocyanin_Actual','Anthocyanin_Predicted', ...
        'TPC_Actual','TPC_Predicted','TFC_Actual','TFC_Predicted', ...
        'DPPH_Actual','DPPH_Predicted'});
    writetable(PredictionTable,fullfile(expFolder,'Independent_Test_Predictions.csv'));
    save(fullfile(expFolder,'Experiment_Results.mat'), ...
        'Results','PredictionTable','Ytrue','Ypred','YpredNorm');
    allMetricTables{expNo}=Results;
    disp(Results);
    clear trainedNet trainInfo lgraph rawPred dsTrain dsVal dsTest opts
end

%% 9. Final combined comparison table
FinalComparison = vertcat(allMetricTables{:});
writetable(FinalComparison,fullfile(outputFolder,'FINAL_ABLATION_COMPARISON.csv'));
save(fullfile(outputFolder,'FINAL_ABLATION_COMPARISON.mat'), ...
    'FinalComparison','experimentNames','targetNames');
fprintf('\nAll four experiments completed. Results are in: %s\n',outputFolder);
disp(FinalComparison);

%% Local function: network builder
function lgraph = buildAblationNetwork(useAttention,multiOutput)
    net = resnet18;
    lgraph = layerGraph(net);
    lgraph = removeLayers(lgraph, ...
        {'fc1000','prob','ClassificationLayer_predictions'});

    if useAttention
        % Retains the user's existing GAP-FC-ReLU-FC-sigmoid channel gate.
        % It is not full CBAM: there is no spatial-attention submodule.
        attentionLayers = [
            globalAveragePooling2dLayer('Name','gap')
            fullyConnectedLayer(512,'Name','fc1')
            reluLayer('Name','relu1')
            fullyConnectedLayer(512,'Name','fc2')
            sigmoidLayer('Name','sigmoid_cbam')];
        lgraph = addLayers(lgraph,attentionLayers);
        lgraph = addLayers(lgraph,multiplicationLayer(2,'Name','cbam_mult'));
        lgraph = connectLayers(lgraph,'pool5','gap');
        % Both inputs to multiplication are essential.
        lgraph = connectLayers(lgraph,'pool5','cbam_mult/in1');
        lgraph = connectLayers(lgraph,'sigmoid_cbam','cbam_mult/in2');
        featureSource = 'cbam_mult';
    else
        featureSource = 'pool5';
    end

    shared = [fullyConnectedLayer(256,'Name','shared_fc')
              reluLayer('Name','shared_relu')];
    lgraph = addLayers(lgraph,shared);
    lgraph = connectLayers(lgraph,featureSource,'shared_fc');

    if multiOutput
        for b=1:4
            branch = [
                fullyConnectedLayer(64,'Name',sprintf('fc_b%d',b))
                reluLayer('Name',sprintf('relu_b%d',b))
                dropoutLayer(0.3,'Name',sprintf('drop_b%d',b))
                fullyConnectedLayer(1,'Name',sprintf('out_b%d',b))];
            lgraph = addLayers(lgraph,branch);
            lgraph = connectLayers(lgraph,'shared_relu',sprintf('fc_b%d',b));
        end
        lgraph = addLayers(lgraph,concatenationLayer(1,4,'Name','concat'));
        lgraph = addLayers(lgraph,regressionLayer('Name','regression_output'));
        for b=1:4
            lgraph = connectLayers(lgraph,sprintf('out_b%d',b), ...
                sprintf('concat/in%d',b));
        end
        lgraph = connectLayers(lgraph,'concat','regression_output');
    else
        singleBranch = [
            fullyConnectedLayer(64,'Name','single_fc')
            reluLayer('Name','single_relu')
            dropoutLayer(0.3,'Name','single_drop')
            fullyConnectedLayer(1,'Name','single_output')
            regressionLayer('Name','regression_output')];
        lgraph = addLayers(lgraph,singleBranch);
        lgraph = connectLayers(lgraph,'shared_relu','single_fc');
    end
end

%% Local function: image preprocessing (same as supplied validation script)
function I = preprocessImage(filename,inputSize)
    I = imread(filename);
    if size(I,3)==1
        I = repmat(I,[1 1 3]);
    elseif size(I,3)>3
        I = I(:,:,1:3);
    end
    I = imresize(I,inputSize(1:2));
    I = im2single(I);
end

%% Local function: common training options
function opts = makeTrainingOptions(dsVal)
    opts = trainingOptions('adam', ...
        'MaxEpochs',50, ...
        'MiniBatchSize',8, ...
        'InitialLearnRate',1e-4, ...
        'Shuffle','every-epoch', ...
        'ValidationData',dsVal, ...
        'ValidationPatience',10, ...
        'Verbose',true, ...
        'Plots','none', ...
        'ExecutionEnvironment','auto');
end
