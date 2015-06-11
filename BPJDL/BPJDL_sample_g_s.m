function g_s = BPJDL_sample_g_s(S,c0,d0,Z,g_s)
%Sample gamma_s
%Version 1: 09/12/2009
%Version 2: 10/21/2009
%Updated in 03/08/2010
%Written by Mingyuan Zhou, Duke ECE, mz1@ee.duke.edu
if length(g_s)==1
    c = c0 + 0.5*numel(Z);   
    d = d0 + 0.5*sum(sum(S.^2)) + 0.5*(numel(Z)-nnz(Z))*(1/g_s);
    g_s = gamrnd(c,1./d);
else
    N = size(S,1);
    c = c0 + 0.5*N;    
    d = d0 + 0.5*sum(S.^2,1) + 0.5*(N-sum(Z,1))./g_s;
    g_s = gamrnd(c,1./d);   
end