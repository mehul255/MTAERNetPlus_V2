function [camMap, score] = gradCAM_MTAER(dlnet, dlImg, targetLayer, outputIndex)

    [score, features] = dlfeval(@modelGradients, dlnet, dlImg, targetLayer, outputIndex);

    gradients = dlgradient(score, features);

    % Global average pooling of gradients
    weights = mean(gradients, [1 2]);

    % Weighted sum
    cam = sum(features .* weights, 3);

    % ReLU and normalize
    camMap = max(extractdata(cam), 0);
    camMap = camMap - min(camMap(:));
    camMap = camMap / (max(camMap(:)) + eps);
end

function [score, features] = modelGradients(dlnet, dlImg, targetLayer, outputIndex)
    % Forward pass with automatic differentiation enabled
    dlY = forward(dlnet, dlImg, 'Outputs', {targetLayer, 'concat'});

    features = dlY{1};  % From attention or intermediate layer
    concatOutput = dlY{2};  % Final concatenated 4-output vector

    score = concatOutput(outputIndex);  % Pick specific output (1–4)
end
