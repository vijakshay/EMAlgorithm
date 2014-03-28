%EM Algorithm for estimating latent class multinomial logit models 
%allowing for multiple observations per individual
%and a class-membership multinomial logit model

clear all

global NP NUMCLASSESIND 
global WEIGHTS WEIGHTSGR 

global CSAV CSCH CH ALTAVMAT ALTCSAV
global NUMVARSF1 NUMVARSF2 NUMVARSF3 
global VARSF1 VARSF2 VARSF3 
global ALTAVMAT1 ALTAVMAT2 ALTAVMAT3 
global ALTAVVEC1 ALTAVVEC2 ALTAVVEC3 

global NUMVARSINDMOD VARSINDMOD ALTAVINDMOD 

global TVARSF TCH TCSCH TALTAV TALTAVMAT TALTAVVEC TCSAV


diary off
delete threeClass01.asc
diary threeClass01.asc


%%%%
%Data Input and Model Specification
%%%%

disp(' ');
disp('MODEL DESCRIPTION: A latent three class model of travel mode choice');
disp('for all tours, mandatory and otherwise, using the BATS 2000 dataset.');
disp('Class 1 denotes mutlimodal all users and Classes 2 and 3 denote');
disp('anti-bike multimodals.');
disp(' ');
disp(' ');

NUMCLASSESIND = 3;

llTol = 1e-06;
maxIters = 10000;
emTol = 1e-04;
nrTol = 1e-12;

disp('Reading data');
data = load('data.txt');

%Class Specific Choice Model 

idp = data(:,1);
idCase = data(:,2);
idAlt = data(:,3);
CH = data(:,4);

x0=data(:,5);
x1=data(:,6);
x2=data(:,7);

VARSF1 = [x0 x1 x2]';
namesFixed1 = {'x0' 'x1' 'x2'};

VARSF2 = [x0 x1 x2]';
namesFixed2 = {'x0' 'x1' 'x2'};

VARSF3 = [x0 x1 x2]';
namesFixed3 = {'x0' 'x1' 'x2'};

clear x0 x1 x2


%Model of Individual Modality Styles 

z0 = data(:,8);
z1 = data(:,9);
z2 = data(:,10);
z3 = data(:,11);
z4 = data(:,12);

varsIndMod = [z0 z1 z2 z3 z4];

namesIndMod = {'z0' 'z1' 'z2' 'z3' 'z4'};

clear z0 z1 z2 z3 z4


%%%%
%Pre-processing 
%%%%

people=unique(idp);
NP=size(unique(idp),1);

%Class-Specific Choice Model

nRows=size(data,1);
ncs=size(unique(idCase),1);

NUMVARSF1=size(VARSF1,1);
NUMVARSF2=size(VARSF2,1);
NUMVARSF3=size(VARSF3,1);

nAltMax=size(unique(idAlt),1);     

xAlt=zeros(nRows,1);
yAlt=zeros(nRows,1);

ALTAVVEC1=zeros(nRows,1);
ALTAVVEC2=zeros(nRows,1);
ALTAVVEC3=zeros(nRows,1);

xCh=zeros(ncs,1);
yCh=zeros(ncs,1);

xCsav=zeros(ncs,1);
yCsav=zeros(ncs,1);

xAltCs=zeros(nRows,1);
yAltCs=zeros(nRows,1);

currentRow=1;
currentCS=1;
counter=1;

svRow=zeros(NUMCLASSESIND+1,1);
svCS=zeros(NUMCLASSESIND+1,1);
svInd=zeros(NUMCLASSESIND+1,1);

for n=1:NP
    
    cs=idCase(idp == people(n));
    yy=CH(idp == people(n));
    aa=idAlt(idp == people(n));    

    tcs=unique(cs);
    ntcs=size(tcs,1);

    for k=1:ntcs %loop over choice situations        

        alt=idAlt(idCase==tcs(k));   

        xCsav(counter,1)=currentCS;
        yCsav(counter,1)=n;

        for j=1:nAltMax
            if sum(alt==j)>0 %check if alternative is available

                xAlt(currentRow,1)=currentRow;                
                yAlt(currentRow,1)=currentCS; 
                
                xAltCs(currentRow,1)=currentRow;
                yAltCs(currentRow,1)=((n-1)*nAltMax) + j;

                ALTAVVEC1(currentRow,1)=1;
                ALTAVVEC2(currentRow,1)=1;                    
                ALTAVVEC3(currentRow,1)=1;                    

                if yy(cs==tcs(k) & aa==j,:)==1;                
                    xCh(counter,1)=currentRow;
                    yCh(counter,1)=currentCS;
                    counter=counter+1;
                end
                currentRow=currentRow+1;
            end
        end              
        currentCS=currentCS+1;
    end   
    
    for r=2:NUMCLASSESIND
        if (n-1) < ((r-1)/NUMCLASSESIND)*NP && n >= ((r-1)/NUMCLASSESIND)*NP
            svRow(r,1)=currentRow-1;
            svCS(r,1)=currentCS-1;
            svInd(r,1)=n;
        end
    end
    
end

svRow(NUMCLASSESIND+1,1)=currentRow-1;
svCS(NUMCLASSESIND+1,1)=currentCS-1;
svInd(NUMCLASSESIND+1,1)=NP;

ALTAVMAT1=sparse(xAlt,yAlt,ALTAVVEC1,nRows,ncs);
ALTAVMAT2=sparse(xAlt,yAlt,ALTAVVEC2,nRows,ncs);
ALTAVMAT3=sparse(xAlt,yAlt,ALTAVVEC3,nRows,ncs);

ALTAVMAT=ALTAVMAT1;

CSCH=sparse(xCh,yCh,ones(size(xCh)),nRows,ncs);
CSAV=sparse(xCsav,yCsav,ones(size(xCsav)),ncs,NP);
ALTCSAV=sparse(xAltCs,yAltCs,ones(size(xAltCs)),nRows,nAltMax*NP);


%Model of Individual Modality Styles 

namesIndMod = ['CSC' namesIndMod];    

NUMVARSINDMOD = (size(varsIndMod,2) + 1) * (NUMCLASSESIND-1);
numVarsPerIndClass = size(varsIndMod,2) + 1;

VARSINDMOD=zeros(NUMVARSINDMOD,NP*NUMCLASSESIND); 

xAlt=zeros(NP*NUMCLASSESIND,1);
yAlt=zeros(NP*NUMCLASSESIND,1);

counter=1;
for n=1:NP  %loop over people    

    if numVarsPerIndClass == 1
        xc = 1;
    else
        xc=[1 mean(varsIndMod(idp == people(n),:),1)];
    end

    for k=1:NUMCLASSESIND-1
        temp=[zeros(k-1,numVarsPerIndClass); xc; zeros(NUMCLASSESIND-1-k,numVarsPerIndClass)];
        VARSINDMOD(:,(n-1)*NUMCLASSESIND+(k+1))=reshape(temp,[NUMVARSINDMOD,1]);
    end

    for k=1:NUMCLASSESIND
        xAlt(counter,1)=counter;
        yAlt(counter,1)=n;
        counter=counter+1;
    end
end

ALTAVINDMOD=sparse(xAlt,yAlt,ones(size(xAlt)),NP*NUMCLASSESIND,NP);


clear aa alt counter cs h hhNumCars indCounter j k yy
clear n nAltMax nRows nalt ncs ntcs tcs temp xc
clear xAlt yAlt xCh yCh xCsav yCsav xAltCs yAltCs
clear varsIndMod
clear currentCS currentRow
clear data 
clear idAlt idCase idp 

%%
%%%%
%Calculate Starting Values
%%%%

param1 = zeros(NUMVARSF1,1);
param2 = zeros(NUMVARSF2,1);
param3 = zeros(NUMVARSF3,1);

paramIndMod = zeros(NUMVARSINDMOD,1);

disp('Calculating starting values');
options=optimset('LargeScale','off','Display','off','GradObj','on','MaxIter',200,...
    'MaxFunEvals',100000,'DerivativeCheck','off');

%Class 1

TVARSF=VARSF1(:,svRow(1,1)+1:svRow(2,1));
TCH=CH(svRow(1,1)+1:svRow(2,1),:); 
TCSCH=CSCH(svRow(1,1)+1:svRow(2,1),svCS(1,1)+1:svCS(2,1));
TALTAV=ALTAVMAT1(svRow(1,1)+1:svRow(2,1),svCS(1,1)+1:svCS(2,1));
TALTAVMAT=ALTAVMAT(svRow(1,1)+1:svRow(2,1),svCS(1,1)+1:svCS(2,1));
TALTAVVEC=ALTAVVEC1(svRow(1,1)+1:svRow(2,1),1); 
TCSAV=CSAV(svCS(1,1)+1:svCS(2,1),svInd(1,1)+1:svInd(2,1));

param1=fminunc(@loglikgr,param1,options);

%Class 2

TVARSF=VARSF2(:,svRow(2,1)+1:svRow(3,1));
TCH=CH(svRow(2,1)+1:svRow(3,1),:); 
TCSCH=CSCH(svRow(2,1)+1:svRow(3,1),svCS(2,1)+1:svCS(3,1));
TALTAV=ALTAVMAT2(svRow(2,1)+1:svRow(3,1),svCS(2,1)+1:svCS(3,1));
TALTAVMAT=ALTAVMAT(svRow(2,1)+1:svRow(3,1),svCS(2,1)+1:svCS(3,1));
TALTAVVEC=ALTAVVEC2(svRow(2,1)+1:svRow(3,1),1); 
TCSAV=CSAV(svCS(2,1)+1:svCS(3,1),svInd(2,1)+1:svInd(3,1));

param2=fminunc(@loglikgr,param2,options);

%Class 3

TVARSF=VARSF3(:,svRow(3,1)+1:svRow(4,1));
TCH=CH(svRow(3,1)+1:svRow(4,1),:); 
TCSCH=CSCH(svRow(3,1)+1:svRow(4,1),svCS(3,1)+1:svCS(4,1));
TALTAV=ALTAVMAT3(svRow(3,1)+1:svRow(4,1),svCS(3,1)+1:svCS(4,1));
TALTAVMAT=ALTAVMAT(svRow(3,1)+1:svRow(4,1),svCS(3,1)+1:svCS(4,1));
TALTAVVEC=ALTAVVEC3(svRow(3,1)+1:svRow(4,1),1); 
TCSAV=CSAV(svCS(3,1)+1:svCS(4,1),svInd(3,1)+1:svInd(4,1));

param3=fminunc(@loglikgr,param3,options);



%%%%
%EM Algorithm
%%%%

disp('Initializing the EM Algorithm');
disp(' ');

converged = 0;
iterCounter = 1;

loglikelihoodOld = 0;

time = zeros(1000,1);
logLikelihood = zeros(1000,1);

options=optimset('LargeScale','off','Display','off','GradObj','on','TolFun',llTol,...
            'MaxIter',maxIters,'MaxFunEvals',10000,'DerivativeCheck','off');

while ~converged
      
    tic;
    
    %%%%
    %Expectaion Step
    %%%%
    
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
    loglikelihoodNew = sum(log(sum(h,1)),2);
    h = h./repmat(sum(h,1),[NUMCLASSESIND,1]);          %h is NUMCLASSES x NP
    

    %Track time       
    
    fmt='%.2d-%.2d-%.2d %.2d:%.2d:%2.0f'; 
    t=datevec(time(iterCounter)/86400); 
    disp(['<' strrep(sprintf(fmt,t),': ',':0') '> Iteration ' num2str(iterCounter-1) ': ' num2str(loglikelihoodNew)]);


    %%%%
    %Maximization Step
    %%%%    
    
    %Class 1
    
    TVARSF = VARSF1;
    TALTAVMAT = ALTAVMAT1;
    TALTAVVEC = ALTAVVEC1;    
    WEIGHTS = h(1,:);                               %WEIGHTS is 1 x NP
    WEIGHTSGR = ALTAVMAT1*CSAV*WEIGHTS';      %WEIGHTSGR is (NUMALT * NCSMAX * NP) x 1
    
    param1=fminunc(@wtCSModel,param1,options);

    %Class 2
    
    TVARSF = VARSF2;
    TALTAVMAT = ALTAVMAT2;
    TALTAVVEC = ALTAVVEC2;    
    WEIGHTS = h(2,:);                               %WEIGHTS is 1 x NP
    WEIGHTSGR = ALTAVMAT2*CSAV*WEIGHTS';      %WEIGHTSGR is (NUMALT * NCSMAX * NP) x 1
    
    param2=fminunc(@wtCSModel,param2,options);

    %Class 3
    
    TVARSF = VARSF3;
    TALTAVMAT = ALTAVMAT3;
    TALTAVVEC = ALTAVVEC3;    
    WEIGHTS = h(3,:);                               %WEIGHTS is 1 x NP
    WEIGHTSGR = ALTAVMAT3*CSAV*WEIGHTS';      %WEIGHTSGR is (NUMALT * NCSMAX * NP) x 1
    
    param3=fminunc(@wtCSModel,param3,options);

    %Class Membership

    WEIGHTS = reshape(h,[NP*NUMCLASSESIND,1]);
    WEIGHTSGR = ALTAVINDMOD*ALTAVINDMOD'*WEIGHTS;
    paramIndMod=fminunc(@wtIndMod,paramIndMod,options);
 
    
    %%%
    %Check for convergence
    %%%
    
    time(iterCounter + 1) = time(iterCounter) + toc;
    logLikelihood(iterCounter) = loglikelihoodNew;

    if abs(loglikelihoodNew - loglikelihoodOld) <= emTol
        converged = 1;
    else
        iterCounter = iterCounter + 1;
    end
    loglikelihoodOld = loglikelihoodNew; 

end

clear WEIGHTS WEIGHTSGR
clear TVARSF TALTAVMAT TALTAVVEC
clear TVARSFOTH TALTAVOTHMAT TALTAVOTHVEC


%%%%
%Newton-Raphson Algorithm
%%%%

disp(' ');
disp('EM Algorithm convergence achieved');
disp('Switching to the Newton Raphson Algorithm');
disp(' ');

param = [param1; param2; param3];
paramStart = [param; paramIndMod];

tic;

options = optimset('LargeScale','off','Display','Iter','GradObj','off','TolFun',nrTol,...
    'MaxFunEvals',10,'DerivativeCheck','off');
[paramFinal,fval,exitflag,output,grad,hessian] = fminunc(@completeloglik,paramStart,options);    

ihess=inv(hessian);
stdErr=sqrt(diag(ihess));

count=1;

param1=zeros(NUMVARSF1,1);
stdErr1=zeros(NUMVARSF1,1);
for i=1:NUMVARSF1
    param1(i,1)=paramFinal(count,1);
    stdErr1(i,1)=stdErr(count,1);
    count=count+1;
end

param2=zeros(NUMVARSF2,1);
stdErr2=zeros(NUMVARSF2,1);
for i=1:NUMVARSF2
    param2(i,1)=paramFinal(count,1);
    stdErr2(i,1)=stdErr(count,1);
    count=count+1;
end

param3=zeros(NUMVARSF3,1);
stdErr3=zeros(NUMVARSF3,1);
for i=1:NUMVARSF3
    param3(i,1)=paramFinal(count,1);
    stdErr3(i,1)=stdErr(count,1);
    count=count+1;
end

paramIndMod=paramFinal(count:end);
stdErrIndMod=stdErr(count:end);


%%%%
%Presentation of Results
%%%%

disp(' ');
disp(['Estimation took ' num2str((time(iterCounter+1) + toc)/60) ' minutes.']);
disp(' ');
disp(['Value of the log-likelihood function at convergence is ' num2str(fval)]);
disp(' ');
disp(' ');

disp('CLASS 1 CHOICE MODEL');    
disp('------------------------------------------------------------------------------ ');
disp('Parameter                                            Est         SE     t-stat');
disp('------------------------------------------------------------------------------ ');
for r=1:NUMVARSF1
    fprintf('%-45s %10.4f %10.4f %10.4f\n', namesFixed1{1,r}, ...
        [param1(r,1) stdErr1(r,1) param1(r,1)./stdErr1(r,1) ]);
end
disp('------------------------------------------------------------------------------ ');
disp(' ');

disp('CLASS 2 CHOICE MODEL');    
disp('------------------------------------------------------------------------------ ');
disp('Parameter                                            Est         SE     t-stat');
disp('------------------------------------------------------------------------------ ');
for r=1:NUMVARSF2
    fprintf('%-45s %10.4f %10.4f %10.4f\n', namesFixed2{1,r}, ...
        [param2(r,1) stdErr2(r,1) param2(r,1)./stdErr2(r,1) ]);
end
disp('------------------------------------------------------------------------------ ');
disp(' ');

disp('CLASS 3 CHOICE MODEL');    
disp('------------------------------------------------------------------------------ ');
disp('Parameter                                            Est         SE     t-stat');
disp('------------------------------------------------------------------------------ ');
for r=1:NUMVARSF3
    fprintf('%-45s %10.4f %10.4f %10.4f\n', namesFixed3{1,r}, ...
        [param3(r,1) stdErr3(r,1) param3(r,1)./stdErr3(r,1) ]);
end
disp('------------------------------------------------------------------------------ ');
disp(' ');

disp('CLASS MEMBERSHIP MODEL');
disp('------------------------------------------------------------------------------ ');
disp('Parameter                                            Est         SE     t-stat');
disp('------------------------------------------------------------------------------ ');
for i=1:numVarsPerIndClass
    for r=2:NUMCLASSESIND
        pos=(i-1)*(NUMCLASSESIND-1)+(r-1);
        fprintf('%-45s %10.4f %10.4f %10.4f\n', strcat(namesIndMod{1,i},' (Class ',num2str(r),')'), ...
            [paramIndMod(pos,1) stdErrIndMod(pos,1) paramIndMod(pos,1)./stdErrIndMod(pos,1) ]);
    end
end
disp('------------------------------------------------------------------------------ ');
disp(' ');

disp(' ' );
disp('You can access the estimated parameters as variable param and paramClass,');
disp('the gradient of the negative of the log-likelihood function as variable grad,');
disp('the hessian of the negative of the log-likelihood function as variable hessian,');
disp('and the inverse of the hessian as variable ihess.');

diary off