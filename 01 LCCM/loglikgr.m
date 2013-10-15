% Calculates log-likelihood function value for multinomial logit model
% This code is input to Matlab's fminunc command
%
% Input param is a column vector of parameters, dimension NUMVARSFx1,
%     containing the fixed coefficients
% Output ll is the scalar value of the negative of the log-likelihood 
%     at the input parameters, and gr is the gradient

function [ll,gr] =loglikgr(param)

global TVARSF TCH TCSCH TALTAV TALTAVMAT TALTAVVEC TCSAV

v=param'*TVARSF;        %v is 1 x (NUMALT * NCSMAX * NP)
ev=exp(v);              %ev is 1 x (NUMALT * NCSMAX * NP)
ev(isinf(ev))=1e+20;    %As precaution when exp(v) is too large for machine
ev=max(ev,1e-300);      %As precaution when exp(v) is too close to zero
nev=ev*TALTAV;          %nev is 1 x (NCSMAX * NP)
nnev=TALTAVMAT*nev';    %nnev is (NUMALT * NCSMAX * NP) x 1
cev=ev'.*TALTAVVEC;     %To account for unavilable alternatives
p=cev./nnev;            %p is (NUMALT * NCSMAX * NP) x 1
p(isnan(p))=1e-300;     %When none of the available alternatives are available
p=max(p,1e-300);        %As a precaution
gr=-(TVARSF*(TCH-p));   %gr is NUMVARSF*1
pcs=p'*TCSCH;           %pcs is 1 x (NCSMAX * NP)
lpcs=log(pcs);          %lpcs is 1 x (NCSMAX * NP)
lpind=lpcs*TCSAV;       %lpcs is 1 x (NP)
ll=-sum(lpind,2);       %ll is 1 x 1