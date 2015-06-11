% =========================================================================
% Example code for Beta Process Joint Dictionary Learning
% 
% Li He
% EECS, University of Tennessee, Knoxville
% Modified for 
% Li He, Hairong Qi, Russell Zaretzki, 
% "Beta Process Joint Dictionary Learning for Coupled Feature Spaces with Application to Single Image Super-Resolution", CVPR 2013
% contact: lhe4@utk.edu
% 10/15/2013
%
% Original version: Jianchao Yang
% ECE Department, University of Illinois at Urbana-Champaign
% =========================================================================

clear; clc; close all;

data_path = 'Data/Training';

dict_size   = 1024;          % dictionary size
patch_size  = 7;            % image patch size  
nSmp        = 100000;       % number of patches to sample
upscale     = 2;            % upscaling factor

% randomly sample image patches
[Xh, Xl] = rnd_smp_patch(data_path, '*.bmp', patch_size, nSmp, upscale);

% delete patches with small variances
[Xh, Xl] = patch_pruning(Xh, Xl, 10);

% joint dictionary learning
[Dh, Dl, M] = dict_learn(Xh, Xl, dict_size);

% save dictionary
dict_path = ['Dictionary/D_' num2str(patch_size) '_' num2str(size(Dh,2)) '_s'...
    num2str(upscale) '.mat' ];

save(dict_path, 'Dh', 'Dl','M');