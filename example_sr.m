% =========================================================================
% Simple demo codes for image super-resolution via sparse representation
% 
% Li He
% EECS, University of Tennessee, Knoxville
% Modified for 
% Li He, Hairong Qi, Russell Zaretzki, 
% "Beta Process Joint Dictionary Learning for Coupled Feature Spaces with Application to Single Image Super-Resolution", CVPR 2013
% contact: lhe4@utk.edu
%
% Original version: Jianchao Yang
% ECE Department, University of Illinois at Urbana-Champaign
% =========================================================================

clear; clc;
warning off;

% load dictionary
dname='D_7_771_s2.mat';
load(['Dictionary/' dname]);


% set parameters
lambda = 0.15;                  % sparsity regularization
overlap = sqrt(size(Dh,1))-1;   % the more overlap the better 
up_scale = 2;                   % scaling factor, depending on the trained dictionary
maxIter = 20;                   % if 0, do not use backprojection

lowdir='Data/low';
highdir='Data/high';
lfile=dir([lowdir filesep '*.bmp']);
hfile=dir([highdir filesep '*.bmp']);

for i=1:numel(lfile)    
    disp(['SR test input image [' lfile(i).name ']']);
    % read test image
    im_l = imread([lowdir filesep lfile(i).name]);

    % change color space, work on illuminance only
    im_l_ycbcr = rgb2ycbcr(im_l);
    im_l_y = im_l_ycbcr(:, :, 1);
    im_l_cb = im_l_ycbcr(:, :, 2);
    im_l_cr = im_l_ycbcr(:, :, 3);

    % image super-resolution based on sparse representation
    [im_h_y] = scbp(im_l_y, up_scale , Dh, Dl, lambda, overlap, M);
    [nrow, ncol] = size(im_h_y);
    % self-similarity 
    [N, ~]       =   Compute_NLM_Matrix( im_h_y , 3);
    NTN          =   N'*N*0.05;
    im_f = sparse(double(im_h_y(:)));
    for j = 1 : 30      
        im_f = im_f  - NTN*im_f;
    end
    im_h_y = reshape(full(im_f), nrow, ncol);
    [im_h_y] = backprojection(im_h_y, im_l_y, maxIter);

    % upscale the chrominance simply by "bicubic" 
    im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
    im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');

    im_h_ycbcr = zeros([nrow, ncol, 3]);
    im_h_ycbcr(:, :, 1) = im_h_y;
    im_h_ycbcr(:, :, 2) = im_h_cb;
    im_h_ycbcr(:, :, 3) = im_h_cr;
    im_h = ycbcr2rgb(uint8(im_h_ycbcr));
    
    % bicubic interpolation for reference
    im_b = imresize(im_l, [nrow, ncol], 'bicubic');

    % read ground truth image
    im = imread([highdir filesep hfile(i).name]);
    im_ycbcr=rgb2ycbcr(im);

    % compute PSNR for the illuminance channel
    bb_rmse = compute_rmse(im, im_b);
    sp_rmse = compute_rmse(im, im_h);
    sp_ssim(i) = ssim(im(:,:,1), im_h(:,:,1));

    bb_psnr(i) = 20*log10(255/bb_rmse);
    sp_psnr(i) = 20*log10(255/sp_rmse);

    fprintf('PSNR for Bicubic Interpolation: %f dB\n', bb_psnr(i));
    fprintf('PSNR for BP-JDL Recovery: %f dB\n', sp_psnr(i));
    fprintf('SSIM for BP-JDL Recovery: %f \n', sp_ssim(i));

    % save images
    imwrite(im_h,['Result' filesep lfile(i).name(1:end-4) '_BPJDL.bmp'],'bmp');
end
