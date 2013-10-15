% Calculates the log-likelihood function value for a latent class mixed logit model
% This code is input to Matlab's fminunc command
%
% Input param is a column vector of parameters, dimension
% (NUMVARSF+NUMVARSR)xNUMCLASSES + NUMARSC
%     containing the fixed coefficients, the parameters of the error
%     components, and the parameters of the class membership model
% Output ll is the scalar value of the negative of the simulated log-likelihood 
%     at the input parameters

function [ll]=completeloglik(param)

global NP NUMCLASSESIND 
global WEIGHTS WEIGHTSGR 

global CSAV CSCH CH ALTAVMAT ALTCSAV
global NUMVARSF1 NUMVARSF2 NUMVARSF3 
global VARSF1 VARSF2 VARSF3 
global ALTAVMAT1 ALTAVMAT2 ALTAVMAT3 
global ALTAVVEC1 ALTAVVEC2 ALTAVVEC3 

global NUMVARSINDMOD VARSINDMOD ALTAVINDMOD 

%Extract Parameters

count=1;

param1=zeros(NUMVARSF1,1);
for i=1:NUMVARSF1
    param1(i,1)=param(count,1);
    count=count+1;
end

param2=zeros(NUMVARSF2,1);
for i=1:NUMVARSF2
    param2(i,1)=param(count,1);
    count=count+1;
end

param3=zeros(NUMVARSF3,1);
for i=1:NUMVARSF3
    param3(i,1)=param(count,1);
    count=count+1;
end

paramIndMod=param(count:end);

%Class 1

v=param1'*VARSF1;         %v is 1 x (NUMALT * NCSMAX * NP)
ev=exp(v);                      %ev is 1 x (NUMALT * NCSMAX * NP)
ev(isinf(ev))=1e+20;            %As precaution when exp(v) is too large for machine
ev=max(ev,1e-300);              %As precaution when exp(v) is too close to zero
nev=ev*ALTAVMAT1;            %nev is 1 x (NCSMAX * NP)
nnev=ALTAVMAT*nev';          %nnev is (NUMALT * NCSMAX * NP) x 1
cev=ev'.*ALTAVVEC1;          %To account for unavilable alternatives
p=cev./nnev;                    %p is (NUMALT * NCSMAX * NP) x 1
p(isnan(p))=1e-300;             %When none of the available alternatives are available
p=max(p,1e-300);                %As a precaution
pcs=p'*CSCH;                 %pcs is 1 x (NCSMAX * NP)
lpcs=log(pcs);                  %lpcs is 1 x (NCSMAX * NP)
lpm=lpcs*CSAV;               %lpcs is 1 x (NP)
pm1=exp(lpm);                   %pm is 1 x (NP)

%Class 2

v=param2'*VARSF2;         %v is 1 x (NUMALT * NCSMAX * NP)
ev=exp(v);                      %ev is 1 x (NUMALT * NCSMAX * NP)
ev(isinf(ev))=1e+20;            %As precaution when exp(v) is too large for machine
ev=max(ev,1e-300);              %As precaution when exp(v) is too close to zero
nev=ev*ALTAVMAT2;            %nev is 1 x (NCSMAX * NP)
nnev=ALTAVMAT*nev';          %nnev is (NUMALT * NCSMAX * NP) x 1
cev=ev'.*ALTAVVEC2;          %To account for unavilable alternatives
p=cev./nnev;                    %p is (NUMALT * NCSMAX * NP) x 1
p(isnan(p))=1e-300;             %When none of the available alternatives are available
p=max(p,1e-300);                %As a precaution
pcs=p'*CSCH;                 %pcs is 1 x (NCSMAX * NP)
lpcs=log(pcs);                  %lpcs is 1 x (NCSMAX * NP)
lpm=lpcs*CSAV;               %lpcs is 1 x (NP)
pm2=exp(lpm);                   %pm is 1 x (NP)

%Class 3

v=param3'*VARSF3;         %v is 1 x (NUMALT * NCSMAX * NP)
ev=exp(v);                      %ev is 1 x (NUMALT * NCSMAX * NP)
ev(isinf(ev))=1e+20;            %As precaution when exp(v) is too large for machine
ev=max(ev,1e-300);              %As precaution when exp(v) is too close to zero
nev=ev*ALTAVMAT3;            %nev is 1 x (NCSMAX * NP)
nnev=ALTAVMAT*nev';          %nnev is (NUMALT * NCSMAX * NP) x 1
cev=ev'.*ALTAVVEC3;          %To account for unavilable alternatives
p=cev./nnev;                    %p is (NUMALT * NCSMAX * NP) x 1
p(isnan(p))=1e-300;             %When none of the available alternatives are available
p=max(p,1e-300);                %As a precaution
pcs=p'*CSCH;                 %pcs is 1 x (NCSMAX * NP)
lpcs=log(pcs);                  %lpcs is 1 x (NCSMAX * NP)
lpm=lpcs*CSAV;               %lpcs is 1 x (NP)
pm3=exp(lpm);                   %pm is 1 x (NP)

probCS=cat(1,pm1,pm2,pm3); %probManMode is NUMCLASSES x NP


%Class Membership

v=paramIndMod'*VARSINDMOD;      %v is 1 x (NP * NUMCLASSESIND)
ev=exp(v);                      %ev is 1 x (NP * NUMCLASSESIND)
ev(isinf(ev))=1e+20;            %As precaution when exp(v) is too large for machine
ev=max(ev,1e-300);              %As precaution when exp(v) is too close to zero
nev=ev*ALTAVINDMOD;             %nev is 1 x NP
nnev=ALTAVINDMOD*nev';          %nnev is (NP * NUMCLASSESIND) x 1
pi=ev./nnev';                   %pi is 1 x (NP * NUMCLASSESIND)

probIndMod=reshape(pi,[NUMCLASSESIND,NP]);          %probIndMod is NUMCLASSES x NP


%Log-likelihood

h = probCS.*probIndMod;           %h is NUMCLASSES x NP
ll = -sum(log(sum(h,1)),2);