
clc;
clear;
close all;

%% STEP 1: LOAD MODEL

load('MTAERNetPlusv2_Final.mat', ...
    'dlnet', 'minVals', 'maxVals');

% Select anthocyanin output
outputIndex = 1;
outputLayer = 'out_b1';

minVal = minVals{1, 'min_Anthocyanin'};
maxVal = maxVals{1, 'max_Anthocyanin'};

%% STEP 2: SELECT FLOWER IMAGE

[file, path] = uigetfile( ...
    {'*.jpg;*.jpeg;*.png'}, ...
    'Select flower image');

if isequal(file, 0)
    return;
end

img = imread(fullfile(path, file));

% Convert to RGB
if size(img,3) == 1
    img = repmat(img,[1 1 3]);
end

if size(img,3) > 3
    img = img(:,:,1:3);
end

%% STEP 3: PREPROCESS IMAGE

inputSize = dlnet.Layers(1).InputSize(1:2);

img = im2double(imresize(img,inputSize));

% Apply same CLAHE preprocessing as original model
for k = 1:3
    img(:,:,k) = adapthisteq(img(:,:,k));
end

%% STEP 4: OUTLINE THE FLOWER

figure;
imshow(img);

title('Draw around the flower, double-click to finish');

roi = drawpolygon('Color','yellow');

mask = createMask(roi);

close;

%% STEP 5: REPLACE BACKGROUND WITH GREY

modifiedImg = img;

for k = 1:3
    channel = modifiedImg(:,:,k);
    channel(~mask) = 0.5;
    modifiedImg(:,:,k) = channel;
end

%% STEP 6: DISPLAY BOTH IMAGES

figure;

subplot(1,2,1);
imshow(img);
title('Original image');

subplot(1,2,2);
imshow(modifiedImg);
title('Grey background');

%% STEP 7: PREPARE MODEL INPUTS

dlOriginal = dlarray(single(reshape(img, ...
    [inputSize 3 1])), 'SSCB');

dlModified = dlarray(single(reshape(modifiedImg, ...
    [inputSize 3 1])), 'SSCB');

%% STEP 8: PREDICT ORIGINAL IMAGE

originalOutput = forward(dlnet, dlOriginal, ...
    'Outputs', outputLayer);

originalNorm = double(gather(extractdata(originalOutput)));

originalPrediction = originalNorm .* ...
    (maxVal-minVal) + minVal;

%% STEP 9: PREDICT MODIFIED IMAGE

modifiedOutput = forward(dlnet, dlModified, ...
    'Outputs', outputLayer);

modifiedNorm = double(gather(extractdata(modifiedOutput)));

modifiedPrediction = modifiedNorm .* ...
    (maxVal-minVal) + minVal;

fprintf('\nNORMALIZED OUTPUT CHECK\n');

fprintf('Original normalized output: %.6f\n', ...
    originalNorm);

fprintf('Modified normalized output: %.6f\n', ...
    modifiedNorm);

fprintf('Training minimum: %.4f\n', minVal);

fprintf('Training maximum: %.4f\n', maxVal);

%% STEP 10: CALCULATE DIFFERENCE

absoluteDifference = abs( ...
    originalPrediction-modifiedPrediction);

percentageChange = absoluteDifference ./ ...
    max(abs(originalPrediction),eps) .* 100;

%% STEP 11: DISPLAY RESULTS

fprintf('\nBACKGROUND DEPENDENCY TEST\n');
fprintf('================================\n');

fprintf('Original prediction: %.4f mg/100g\n', ...
    originalPrediction);

fprintf('Grey background prediction: %.4f mg/100g\n', ...
    modifiedPrediction);

fprintf('Absolute difference: %.4f mg/100g\n', ...
    absoluteDifference);

fprintf('Percentage change: %.2f %%\n', ...
    percentageChange);

%% STEP 12: SAVE RESULTS

Result = table(originalPrediction, ...
    modifiedPrediction, ...
    absoluteDifference, ...
    percentageChange, ...
    'VariableNames', ...
    {'Original','GreyBackground', ...
    'AbsoluteDifference','PercentageChange'});

writetable(Result,'Background_Test_Result.xlsx');

fprintf('\nResults saved successfully.\n');
