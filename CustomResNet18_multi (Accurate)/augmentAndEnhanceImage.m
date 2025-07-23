function out = augmentAndEnhanceImage(file, targetSize)
    img = imread(file);
    img = im2double(imresize(img, targetSize));
    for i = 1:3
        img(:,:,i) = adapthisteq(img(:,:,i));
    end
    if rand > 0.5
        img = imrotate(img, randi([-10,10]), 'bilinear','crop');
    end
    if rand > 0.5
        factor = 0.85 + 0.3*rand();
        img = min(img * factor, 1.0);
    end
    if rand > 0.5
        img = imtranslate(img, [randi([-5 5]) randi([-5 5])]);
    end
    out = im2single(img);
end
