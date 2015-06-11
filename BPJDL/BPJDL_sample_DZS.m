% =========================================================================
% Beta Process Joint Dictionary Learning Gibbs sampling
% 
% Li He
% EECS, University of Tennessee, Knoxville
% Modified for 
% Li He, Hairong Qi, Russell Zaretzki, 
% "Beta Process Joint Dictionary Learning for Coupled Feature Spaces with Application to Single Image Super-Resolution", CVPR 2013
% contact: lhe4@utk.edu
% 10/15/2013
%
% Original version: by Mingyuan Zhou, Duke ECE, mz1@ee.duke.edu
% =========================================================================
function [Xh_k, Xl_k, Dh, Dl, Sh, Sl, Z] = BPJDL_sample_DZS(Xh_k, Xl_k, Dh, Dl, Sh, Sl, Z, Pi, g_sh, g_sl, g_epsh, g_epsl, Dsample, Zsample, Ssample);


[Ph,N] = size(Xh_k);
[Pl,N] = size(Xl_k);
K = size(Dh,2);

g_sh = repmat(g_sh,1,K);
g_sl = repmat(g_sl,1,K);

for k=1:K
    nnzk = nnz(Z(:,k));
    if nnzk>0
        Xh_k(:,Z(:,k)) = Xh_k(:,Z(:,k)) + Dh(:,k)*Sh(Z(:,k),k)';    
        Xl_k(:,Z(:,k)) = Xl_k(:,Z(:,k)) + Dl(:,k)*Sl(Z(:,k),k)'; 
    end
    
    if Dsample
        %Sample Dh
        sig_Dkh = 1./(g_epsh*sum(Sh(Z(:,k),k).^2)+Ph);
        mu_Dh = g_epsh*sig_Dkh* (Xh_k(:,Z(:,k))*Sh(Z(:,k),k));
        Dh(:,k) = mu_Dh + randn(Ph,1)*sqrt(sig_Dkh);      
        %Sample Dl
        sig_Dkl = 1./(g_epsl*sum(Sl(Z(:,k),k).^2)+Pl);
        mu_Dl = g_epsl*sig_Dkl* (Xl_k(:,Z(:,k))*Sl(Z(:,k),k));
        Dl(:,k) = mu_Dl + randn(Pl,1)*sqrt(sig_Dkl);             
    end
    
    if Zsample || Ssample
        DTDh = sum(Dh(:,k).^2);
        DTDl = sum(Dl(:,k).^2);
    end
    
    if Zsample
        %Sample Z        
        Skh = full(Sh(:,k));
        Skl = full(Sl(:,k));
        %draw the Pesudo Weights S(i,k) from prior if Z(i,k)=0
        Skh(~Z(:,k)) = randn(N-nnz(Z(:,k)),1)*sqrt(1/g_sh(k));
        Skl(~Z(:,k)) = randn(N-nnz(Z(:,k)),1)*sqrt(1/g_sl(k));
        temp = exp( - 0.5*g_epsh*( (Skh.^2 )*DTDh - 2*Skh.*(Xh_k'*Dh(:,k)) )...
            - 0.5*g_epsl*( (Skl.^2 )*DTDl - 2*Skl.*(Xl_k'*Dl(:,k)) )).*Pi(:,k);
        %temp = exp(temp).*Pi(:,k);
        Z(:,k) = sparse( rand(N,1) > ((1-Pi(:,k))./(temp+1-Pi(:,k))) );
    end
    
    nnzk = nnz(Z(:,k));    
    if Ssample
        if nnzk>0
            %Sample Sh
            sigS1h = 1/(g_sh(k) + g_epsh*DTDh);
            Sh(:,k) = sparse(find(Z(:,k)),1,randn(nnz(Z(:,k)),1)*sqrt(sigS1h)+ sigS1h*(g_epsh*(Xh_k(:,Z(:,k))'*Dh(:,k))),N,1);        
            %Sample Sl
            sigS1l = 1/(g_sl(k) + g_epsl*DTDl);
            Sl(:,k) = sparse(find(Z(:,k)),1,randn(nnz(Z(:,k)),1)*sqrt(sigS1l)+ sigS1l*(g_epsl*(Xl_k(:,Z(:,k))'*Dl(:,k))),N,1);        
        else
            Sh(:,k) = 0;
            Sl(:,k) = 0;
        end
    end
    
    if nnzk>0
        Xh_k(:,Z(:,k)) = Xh_k(:,Z(:,k))- Dh(:,k)*Sh(Z(:,k),k)';
        Xl_k(:,Z(:,k)) = Xl_k(:,Z(:,k))- Dl(:,k)*Sl(Z(:,k),k)';
    end
end

end

