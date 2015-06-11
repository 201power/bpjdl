function Pi = BPJDL_sample_Pi(Z,a0,b0)
%Sample Pi
%Version 1: 09/12/2009
%Updated in 03/08/2010
%Written by Mingyuan Zhou, Duke ECE, mz1@ee.duke.edu
sumZ = full(sum(Z,1));
N = size(Z,1);
Pi = betarnd(sumZ+a0, b0+N-sumZ);
end