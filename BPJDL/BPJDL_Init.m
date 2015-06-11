function [D,S,g_eps,g_s] = BPJDL_Init(X_k,pars)
%Initialization
%10/11/2012
%Li He, UTK EECS, lhe4@utk.edu

K=pars.K;
[P,N]=size(X_k);

g_eps = 1/((25/255)^2);

g_s = 1;

if strcmp(pars.InitOption,'SVD')==1
    [U_1,S_1,V_1] = svd(full(X_k),'econ');
    if P<=K
        D = zeros(P,K);
        D(:,1:P) = U_1*S_1;
        S = zeros(N,K);
        S(:,1:P) = V_1;
    else
        D =  U_1*S_1;
        D = D(1:P,1:K);
        S = V_1;
        S = S(1:N,1:K);
    end
else
	% random initialization
    D = rand(P,K)-0.5;
    D = D - repmat(mean(D,1), size(D,1),1);
    D = D*diag(1./sqrt(sum(D.*D)));

    S = randn(N,K);
end

end





