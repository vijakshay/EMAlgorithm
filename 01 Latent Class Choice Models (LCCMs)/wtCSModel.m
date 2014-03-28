% Calculates the weighted log-likelihood function value for mixed logit model
% This code is input to Matlab's fminunc command
%
% Input param is a column vector of parameters, dimension (NUMVARSF+NUMVARSR)x1
%     containing the fixed coefficients, and the parameters of the error components
% Output ll is the scalar value of the negative of the simulated log-likelihood 
%     at the input parameters, and g is the gradient

function [ll,gr] =wtCSModel(param)

global TVARSF TALTAVMAT TALTAVVEC
global ALTAVMAT CH CSCH CSAV 
global WEIGHTSGR WEIGHTS

v=param'*TVARSF;             %v is 1 x (NUMALT * NCSMAX * NP)
ev=exp(v);                      %ev is 1 x (NUMALT * NH)
ev(isinf(ev))=1e+20;            %As precaution when exp(v) is too large for machine
ev=max(ev,1e-300);              %As precaution when exp(v) is too close to zero
nev=ev*TALTAVMAT;            %nev is 1 x (NCSMAX * NP)
nnev=ALTAVMAT*nev';          %nnev is (NUMALT * NCSMAX * NP) x 1
cev=ev'.*TALTAVVEC;          %To account for unavilable alternatives
p=cev./nnev;                    %p is (NUMALT * NCSMAX * NP) x 1
p(isnan(p))=1e-300;             %When none of the available alternatives are available
p=max(p,1e-300);                %As a precaution

gr=WEIGHTSGR.*(CH-p);        %gr is (NUMALT * NCSMAX * NP) x 1
gr=-(TVARSF*gr);             %gr is NUMVARSF*1

pcs=p'*CSCH;                 %pcs is 1 x (NCSMAX * NP)
lpcs=log(pcs);                  %lpcs is 1 x (NCSMAX * NP)
lp=lpcs*CSAV;                %lpcs is 1 x (NP)
ll=-sum(lp.*WEIGHTS,2);         %ll is 1 x 1
