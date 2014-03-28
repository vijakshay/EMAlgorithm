% Calculates log-likelihood function value for multinomial logit model
% This code is input to Matlab's fminunc command
%
% Input param is a column vector of parameters, dimension (NUMVARSF+NUMVARSR)x1
%     containing the fixed coefficients, and the parameters of the error components
% Output ll is the scalar value of the negative of the simulated log-likelihood 
%     at the input parameters, and gg is the gradient

function [ll,g] = wtIndMod(param)

global VARSINDMOD ALTAVINDMOD 
global WEIGHTS WEIGHTSGR

v=param'*VARSINDMOD;            %v is 1 x (NP * NUMCLASSESIND)
ev=exp(v);                      %ev is 1 x (NP * NUMCLASSESIND)
ev(isinf(ev))=1e+20;            %As precaution when exp(v) is too large for machine
nev=ev*ALTAVINDMOD;             %nev is 1 x (NP)
nnev=ALTAVINDMOD*nev';          %nnev is (NP * NUMCLASSESIND) x 1
p=ev'./nnev;                    %p is (NP * NUMCLASSESIND) x 1
p=max(p,1e-300);                %As precaution when exp(v) is too close to zero

g=p.*WEIGHTSGR;                 %g is (NP * NUMCLASSESIND) x 1
g=-(VARSINDMOD*(WEIGHTS-g));    %g is NUMVARSF x 1

pw=log(p).*WEIGHTS;             %pw is (NP * NUMCLASSESIND) x 1
ll=-sum(pw,1);                  %ll is 1 x 1
