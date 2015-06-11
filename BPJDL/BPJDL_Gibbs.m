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
% =========================================================================

function [D,M] = BPJDL_Gibbs(Xh_k,Xl_k,pars)
    % record training time
    tStart=tic;
    
    K=pars.K;
    % Set Hyperparameters
    c0=1e-6;% parameters for gamma_s (variances of s)
    d0=1e-6;
    e0=1e-6; % parameters for gamma_epsilon (variance of x)
    f0=1e-6;
    
    [Ph,N] = size(Xh_k);
    [Pl,N] = size(Xl_k);
    %Sparsity Priors
    if strcmp(pars.InitOption,'SVD')==1
        a0=1;
        b0=N/8;
    else
        a0=1;
        b0=1;
    end
    
    [Dh,Sh,g_epsh,g_sh] = BPJDL_Init(Xh_k,pars);
    [Dl,Sl,g_epsl,g_sl] = BPJDL_Init(Xl_k,pars);
    g_epsh = 1/(pars.ratioh*var(Xh_k(:)));
    g_epsl = 1/(pars.ratiol*var(Xl_k(:)));
    % init Z;
    Z = logical(sparse(N,K));
    Sh=Sh.*Z;Sl=Sl.*Z;
    % init Pi;
    Pi = 0.01*ones(1,K);
    Pi = Pi-Pi+0.001;   
    
    Xh_k = Xh_k - Dh*Sh';
    Xl_k = Xl_k - Dl*Sl';
    
    ah=zeros(size(Sh));al=zeros(size(Sl));
    M=[];Z_avg=[];
    D_avg=zeros(Ph+Pl,K);
    K_all=[];
    time=0;avg_count=0;

    disp(['Number of observations: ' num2str(N)]);
    disp(['Start BPJDL Gibbs sampling....burnin [' num2str(pars.burnin) ']']);

    for iter=1:pars.MaxIter
        istart=tic;
        % sample Dh, Dl, Sh, Sl, Z
        [Xh_k, Xl_k, Dh, Dl, Sh, Sl, Z] = BPJDL_sample_DZS(Xh_k, Xl_k, Dh, Dl, Sh, Sl, Z, Pi, g_sh, g_sl, g_epsh, g_epsl, true, true , true);
        % sample Pi
        Pi = BPJDL_sample_Pi(Z,a0,b0);
        % 1/g_s: variance for s,
        g_sh = BPJDL_sample_g_s(Sh,c0,d0,Z,g_sh);
        g_sl = BPJDL_sample_g_s(Sl,c0,d0,Z,g_sl);
        % g_eps: noise var, LH: fix g_eps for single image super-resolution
        % application
        %g_epsh = BPJDL_sample_g_eps(Xh_k,e0,f0);
        %g_epsl = BPJDL_sample_g_eps(Xl_k,e0,f0);

        Z_avg(iter)=full(mean(sum(Z,2)));
        ittime=toc(istart);

        nstd_h(iter) = sqrt(1/g_epsh);
        nstd_l(iter) = sqrt(1/g_epsl);
        time=time+ittime;
        errh(iter)=sqrt(sum(sum(Xh_k.^2))/N);
        errl(iter)=sqrt(sum(sum(Xl_k.^2))/N);
        err(iter)=sqrt((errh(iter).^2*N+errl(iter).^2*N)/N);
        if pars.ReduceDictSize && iter>10
            sumZ = sum(Z,1)';
            if min(sumZ)==0
                Pidex = sumZ==0;
                Dh(:,Pidex)=[];
                Dl(:,Pidex)=[];
                D_avg(:,Pidex)=[];
                K = size(Dh,2);
                Z(:,Pidex)=[];Pi(Pidex)=[];
                Sh(:,Pidex)=[];Sl(:,Pidex)=[];
                if (~isempty(ah)) ah(:,Pidex)=[];al(:,Pidex)=[]; end
                if (~isempty(M)) M(:,Pidex)=[];M(Pidex,:)=[]; end
            end
        end

        if (iter>pars.burnin)
            D=[Dh;Dl];
            D_avg=D_avg+D;
            ah=ah+Sh.*Z;al=al+Sl.*Z;
            avg_count=avg_count+1;
        end % end of burnin

        K_all(iter)=K;

        disp(['iter:', num2str(iter), ' time:', num2str(ittime,2), 's Z:', num2str(full(mean(sum(Z,2))),3),...
            ' K:', num2str(K),' RMSE:',num2str(err(iter),2)]);

    end    % end of sample

    % average samples 
    D_avg=D_avg/avg_count;
    ah=ah/avg_count;al=al/avg_count;
    M=(al\ah)';
    disp(['Dictonary averaged over ' num2str(avg_count) ' samples.']);
    D=D_avg; % return averaged dictionary elements

    train_time=toc(tStart);
    
    disp(['Total time: ' num2str(train_time/3600) ' hours']);
    disp(['Learning RMS: ' num2str(err(iter))]);
    disp(['Sparsity: ' num2str(full(mean(sum(Z,2)))/K_all(iter)*100) '%']);

end