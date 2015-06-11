function g_eps = BPJDL_sample_g_eps(X_k,e0,f0,Yflag)
%Sample g_eps
%Version 1: 09/12/2009
%Version 2: 10/26/2009
%Written by Mingyuan Zhou, Duke ECE, mz1@ee.duke.edu
if nargin<4
    sumYflag = numel(X_k);
else
    sumYflag = nnz(Yflag);
end
e = e0 + 0.5*sumYflag;
f = f0 + 0.5*sum(sum((X_k).^2));
g_eps = gamrnd(e,1./f);
end