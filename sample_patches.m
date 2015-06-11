function [HP, LP] = sample_patches(im, patch_size, patch_num, upscale)

addpath('Dictionary');
R_thresh=0.27;

if size(im, 3) == 3,
    hIm = rgb2gray(im);
else
    hIm = im;
    disp(['grayscale image sampled']);
end

% generate low resolution counter parts
lIm = imresize(hIm, 1/upscale, 'bicubic');
lIm = imresize(lIm, size(hIm), 'bicubic');
[nrow, ncol] = size(hIm);

x = randperm(nrow-2*patch_size-1) + patch_size;
y = randperm(ncol-2*patch_size-1) + patch_size;

[X,Y] = meshgrid(x,y);

xrow = X(:);
ycol = Y(:);

hIm = double(hIm);
lIm = double(lIm);

H = zeros(patch_size^2,     length(xrow));
L = zeros(4*patch_size^2,   length(xrow));
 
% compute the first and second order gradients
hf1 = [-1,0,1];
vf1 = [-1,0,1]';
 
lImG11 = conv2(lIm, hf1,'same');
lImG12 = conv2(lIm, vf1,'same');
 
hf2 = [1,0,-2,0,1];
vf2 = [1,0,-2,0,1]';
 
lImG21 = conv2(lIm,hf2,'same');
lImG22 = conv2(lIm,vf2,'same');

idx=1;ii=1;n=length(xrow);
while (idx < patch_num) && (ii<=n),    
    row = xrow(ii);
    col = ycol(ii);
    
    Hpatch = hIm(row:row+patch_size-1,col:col+patch_size-1);
    
    Lpatch1 = lImG11(row:row+patch_size-1,col:col+patch_size-1);
    Lpatch2 = lImG12(row:row+patch_size-1,col:col+patch_size-1);
    Lpatch3 = lImG21(row:row+patch_size-1,col:col+patch_size-1);
    Lpatch4 = lImG22(row:row+patch_size-1,col:col+patch_size-1);
        
    Lpatch = [Lpatch1(:),Lpatch2(:),Lpatch3(:),Lpatch4(:)];
    Lpatch = Lpatch(:);
     
    % check if it is a stochastic patch
    H=Hpatch(:)-mean(Hpatch(:));
    Hnorm=sqrt(sum(H.^2));
    H_normalised=reshape(H/Hnorm,patch_size,patch_size);
    % eliminate that small variance patch
    if var(H)>10
        % eliminate stochastic patch
        if dominant_measure(H_normalised)>R_thresh
        %if dominant_measure_G(Lpatch1,Lpatch2)>R_thresh
            HP(:,idx) = H;
            LP(:,idx) = Lpatch;
            idx=idx+1;
        end
    end
    
    ii=ii+1;
end

fprintf('sampled %d patches.\r\n',patch_num);
end

function R = dominant_measure(p)
% calculate the dominant measure
% ref paper: Eigenvalues and condition numbers of random matries, 1988
% p = size n x n patch

hf1 = [-1,0,1];
vf1 = [-1,0,1]';
Gx = conv2(p, hf1,'same');
Gy = conv2(p, vf1,'same');

G=[Gx(:),Gy(:)];
[U S V]=svd(G);

R=(S(1,1)-S(2,2))/(S(1,1)+S(2,2));

end
