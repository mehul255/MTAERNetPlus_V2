
function features = extractRGBFeatures(file)

    I = imread(file);

    if size(I,3) == 1
        I = repmat(I,1,1,3);
    elseif size(I,3) == 4
        I = I(:,:,1:3);
    end

    I = im2double(imresize(I,[224 224]));

    R = I(:,:,1);
    G = I(:,:,2);
    B = I(:,:,3);

    meanR = mean(R(:));
    meanG = mean(G(:));
    meanB = mean(B(:));

    totalRGB = meanR + meanG + meanB + eps;

    r = meanR/totalRGB;
    g = meanG/totalRGB;

    HSV = rgb2hsv(I);

    H = mean(HSV(:,:,1),'all');
    S = mean(HSV(:,:,2),'all');
    V = mean(HSV(:,:,3),'all');

    ExG = mean(2*G-R-B,'all');
    ExR = mean(1.4*R-G,'all');

    features = [meanR meanG meanB r g H S V ExG ExR];

end
