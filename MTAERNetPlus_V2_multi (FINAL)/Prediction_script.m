clc; clear;

%% Load trained model and scalers
data = load('MTAERNetPlusv2_Final.mat');  % Contains trainedNet, minVals, maxVals
net = data.trainedNet;
minVals = data.minVals;
maxVals = data.maxVals;

targetNames = ["Anthocyanin", "TPC", "TFC", "DPPH"];

%% Select and preprocess image
[filename, pathname] = uigetfile({'*.jpg;*.png'}, 'Select flower image');
if isequal(filename, 0)
    disp('No file selected.');
    return;
end

imgPath = fullfile(pathname, filename);
img = imread(imgPath);
img = imresize(img, [224 224]);
img = im2single(img);
dlImg = reshape(img, [224 224 3 1]);

%% Predict normalized outputs
predNorm = predict(net, dlImg);  % 1x4 vector

%% Denormalize
truePred = zeros(1, 4);
for i = 1:4
    minVal = minVals{1, "min_" + targetNames(i)};
    maxVal = maxVals{1, "max_" + targetNames(i)};
    truePred(i) = predNorm(i) * (maxVal - minVal) + minVal;
end

%% Display Results
fprintf('\n📷 Prediction for: %s\n', filename);
for i = 1:4
    fprintf('%s: %.2f\n', targetNames(i), truePred(i));
end
