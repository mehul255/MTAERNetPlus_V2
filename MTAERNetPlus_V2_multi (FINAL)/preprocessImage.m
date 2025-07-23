function out = preprocessImage(file, sz)
    img = imread(file);
    img = im2double(imresize(img, sz(1:2)));
    for i = 1:3
        img(:,:,i) = adapthisteq(img(:,:,i));
    end
    out = im2single(img);
end
