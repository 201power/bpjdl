% =========================================================================
% Example code for Beta Process Joint Dictionary Learning
% Dictionary learning function 
%
% Li He
% EECS, University of Tennessee, Knoxville
% Li He, Hairong Qi, Russell Zaretzki, 
% "Beta Process Joint Dictionary Learning for Coupled Feature Spaces with Application to Single Image Super-Resolution", CVPR 2013
% contact: lhe4@utk.edu
%
% 10/15/2013
% =========================================================================

function [Dh, Dl, M] = dict_learn(Xh, Xl, dict_size)

    addpath('BPJDL');

    hDim = size(Xh, 1);
    lDim = size(Xl, 1);

    % normalize Xh and Xl !
    hNorm = sqrt(sum(Xh.^2));
    lNorm = sqrt(sum(Xl.^2));

    Idx = find( hNorm & lNorm );
    Xh = Xh(:, Idx);
    Xl = Xl(:, Idx);

    Xh = Xh./repmat(sqrt(sum(Xh.^2, 1)), hDim, 1);
    Xl = Xl./repmat(sqrt(sum(Xl.^2, 1)), lDim, 1);
    
    % BP-JDL parameters
    pars.K=dict_size;
    pars.ReduceDictSize = true; %Reduce the ditionary size during training if it is TRUE, can be used to reduce computational complexity 
    pars.InitOption = 'Rand'; %Initialization with 'SVD' or 'Rand'
    pars.MaxIter=10000; % number of samples
    pars.burnin=9500; % burnin
    pars.ratioh=0.25; % ratio of the noise variance to data variance (high-res data)
    pars.ratiol=0.25 % ratio of the noise variance to data variance (low-res data)

    % dictionary learning 
    [D,M] = BPJDL_Gibbs(Xh,Xl,pars);
        
    Dh = D(1:hDim, :);
    Dl = D(hDim+1:end, :);
    
end
