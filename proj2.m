%%%%%% Data Read %%%%%%
fid = fopen('Querylevelnorm.txt','rt');
tmp = textscan(fid,'%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s',69623);
y = tmp{1,1};    
parameters = tmp(:,3:48);
X = [parameters{:}];
featureArr = zeros(69623,46);
[rowSize,columnSize] = size(X);
for k=1:rowSize
    for j=1:columnSize
        strings = strsplit(X{k,j},':');
        featureArr(k,j) = str2double(strings(2));
    end    
end
relArr = str2double(Y);

fclose(fid);

%%%%%%% Real Data Set %%%%%%%
N1 = floor(0.8 * length(featureArr));
N2 = floor(0.9 * length(featureArr));

randFeatureArr = randperm(length(featureArr));
randFeatureArr = (randFeatureArr).';

trainIndexes = randFeatureArr(1:N1);
trainFeatureArr = featureArr(trainIndexes,:);
trainRelArr = relArr(trainIndexes,:);

validationIndexes = randFeatureArr(N1+1:N2);
valFeatureArr = relArr(validationIndexes,:);
valRelArr = relArr(validationIndexes,:);

testIndexes = randFeatureArr(N2+1:end);
testFeatureArr = featureArr(testIndexes,:);
testRelArr = relArr(testIndexes,:);

N2 = N2 - N1;
trainInd1 = trainIndexes;
validInd1 = validationIndexes;

M1 = 30;
D=size(featureArr,2);

%mu-random
a=floor(N1/M1);
mu1=[];
lower=1;
upper=a;
for i=1:M1
	rand_num=randi([lower,upper],1,1);
	mu1=horzcat(mu1,trainFeatureArr(rand_num,:).');
	lower=upper+1;
	upper=a*(i+1);
end

sigma = 0.1 * var(trainFeatureArr);
%sigma = var(trainFeatureArr);

for i=1:D
    if sigma(1,i)<0.0001
        sigma(1,i)=0.01;
    end    
end    
%sigma(~sigma) = 0.1;
sigma = diag(sigma);
Sigma1 = zeros(D,D,M1);
for i=1:M1
   Sigma1(:,:,i) = sigma;
end

phi = zeros(N1,M1);
phi(:,1) = 1;

for i = 1:N1
    for j = 2:M1
        phi(i,j) = exp (-0.5 * (trainFeatureArr(i,:).'-mu1(:,j)).' * inv(Sigma1(:,:,j)) * (trainFeatureArr(i,:).'-mu1(:,j)) );
    end
end 

%minW1= zeros(M1,1);
%minValError=100000;
%minLambda1=0;
%lambda1 = 0;
%minPhi2=zeros(N2,M1);


phi2 = zeros(N2,M1);
phi2(:,1) = 1;

for i = 1:N2
	for j = 2:M1
		phi2(i,j) = exp (-0.5 * (valFeatureArr(i,:).'-mu1(:,j)).' * inv(Sigma1(:,:,j)) * (valFeatureArr(i,:).'-mu1(:,j)) );
	end
end 


%for k=1:10
%	lambda1=lambda1+0.05;
	w1 =  inv((lambda1 * eye(M1))+(phi.'*phi)) * (phi.' * trainRelArr); 

	% Calculating phi for X Validation and Y Validation
	
	lambda1 = 0.5;
	
	validationError = 0.5 * (valRelArr - (phi2 * w1)).' * (valRelArr - (phi2 * w1));  
	validPer1 = sqrt(((2 * validationError) / N2));
	
%	if(validPer1<minValError)
%		minW1=w1;
%		minValError=validPer1;
%		minLambda1=lambda1
		%minPhi2=phi2;
	end
	
%end

%w1=minW1;
%validPer1=minValError;
%lambda1=minLambda1;

trainingError = 0.5 * (trainRelArr - (phi * w1)).' * (trainRelArr - (phi * w1));  
trainPer1 = sqrt(((2 * trainingError) / N1));

n=length (testFeatureArr);
phi_test = zeros(n,M1);
phi_test(:,1) = 1;

for i = 1:n
	for j = 2:M1
		phi_test(i,j) = exp (-0.5 * (testFeatureArr(i,:).'-mu1(:,j)).' * inv(Sigma1(:,:,j)) * (testFeatureArr(i,:).'-mu1(:,j)) );
	end
end
testError = 0.5 * (testRelArr - (phi_test * w1)).' * (testRelArr - (phi_test * w1));  
testPer1 = sqrt(((2 * testError) / n));


%%%%%%%%%% Synthetic Data %%%%%%%%%

featureArrSyn=x.';
relArrSyn=t;
N3 = floor(0.8 * length(featureArrSyn));
N4 = floor(0.9 * length(featureArrSyn));

randFeatureArr = randperm(length(featureArrSyn));
randFeatureArr = (randFeatureArr).';

trainingIndexes = randFeatureArr(1:N3);
trainFeatureArrSyn = featureArrSyn(trainingIndexes,:);
trainRelArrSyn = relArrSyn(trainingIndexes,:);

validationIndexes = randFeatureArr(N3+1:N4);
valFeatureArrSyn = featureArrSyn(validationIndexes,:);
valRelArrSyn = relArrSyn(validationIndexes,:);

testIndexes = randFeatureArr(N4+1:end);
testFeatureArrSyn = featureArrSyn(testIndexes,:);
testRelArrSyn = relArrSyn(testIndexes,:);

N4 = N4 - N3;
trainInd2 = trainingIndexes;
validInd2 = validationIndexes;

M2 = 5;
D=size(featureArrSyn,2);

%mu-random
b=floor(N3/M2);
mu2=[];
lower=1;
upper=b;
for i=1:M2
	rand_num=randi([lower,upper],1,1);
	mu2=horzcat(mu2,trainFeatureArrSyn(rand_num,:).');
	lower=upper+1;
	upper=b*(i+1);
end

sigma = 0.1 * var(trainFeatureArrSyn);
%sigma = var(trainFeatureArrSyn);

for i=1:D
    if sigma(1,i)<0.0001
        sigma(1,i)=0.01;
    end    
end    
%sigma(~sigma) = 0.1;
sigma = diag(sigma);
Sigma2 = zeros(D,D,M2);
for i=1:M2
   Sigma2(:,:,i) = sigma;
end

phi3 = zeros(N3,M2);
phi3(:,1) = 1;

for i = 1:N3
    for j = 2:M2
        phi3(i,j) = exp (-0.5 * (trainFeatureArrSyn(i,:).'-mu2(:,j)).' * inv(Sigma2(:,:,j)) * (trainFeatureArrSyn(i,:).'-mu2(:,j)) );
    end
end 


phi4 = zeros(N4,M2);
phi4(:,1) = 1;

for i = 1:N4
	for j = 2:M2
		phi4(i,j) = exp (-0.5 * (valFeatureArrSyn(i,:).'-mu2(:,j)).' * inv(Sigma2(:,:,j)) * (valFeatureArrSyn(i,:).'-mu2(:,j)) );
	end
end 


%minW2= zeros(M1,1);
%minValError=100000;
%minLambda2=0;
%lambda2 = 0;
%minPhi4=zeros(N4,M1);

%for k=1:10
%	lambda2=lambda2+0.1;
	lambda2 = 0.1;
	w2 =  inv((lambda2 * eye(M2))+(phi3.'*phi3)) * (phi3.' * trainRelArrSyn); 

	
	% Calculating phi3 for X Validation and Y Validation
	%N4 = N4 - N3;

	%lambda2 = 0.5;
	%wvalidation =  inv((lambda2 * eye(M2,M2))+(phi4.'*phi4)) * (phi4.' * valRelArrSyn); 

	validationError = 0.5 * (valRelArrSyn - (phi4 * w2)).' * (valRelArrSyn - (phi4 * w2));  
	validPer2 = sqrt(((2 * validationError) / N4));

%	if(validPer2<minValError)
%		minW2=w2;
%		minValError=validPer2;
%		minLambda2=lambda2
		%minPhi4=phi4;
	end
%end

%w2=minW2;
%validPer2=minValError;
%lambda2=minLambda2;
%phi4=minPhi4;


trainingError = 0.5 * (trainRelArrSyn - (phi3 * w2)).' * (trainRelArrSyn - (phi3 * w2));  
trainPer2 = sqrt(((2 * trainingError) / N3));

n1=length (testFeatureArrSyn);
phi_test2 = zeros(n1,M2);
phi_test2(:,1) = 1;

for i = 1:n1
	for j = 2:M2
		phi_test2(i,j) = exp (-0.5 * (testFeatureArrSyn(i,:).'-mu2(:,j)).' * inv(Sigma2(:,:,j)) * (testFeatureArrSyn(i,:).'-mu2(:,j)) );
	end
end
testError2 = 0.5 * (testRelArrSyn - (phi_test2 * w2)).' * (testRelArrSyn - (phi_test2 * w2));  
testPer2 = sqrt(((2 * testError2) / n1));


%%%%%%%%%% SGD - Real Data Set %%%%%%%%%
%SGD

w01(1:M1,1) = 3;
%w01= 400+4*rand(M1,1);
%w01= w1+10;
eta1=1;
prevW=w01;

deltaEW=prevW;
deltaED=-((trainRelArr(1,:) - prevW.'*phi(1,:).')*phi(1,:)).';
deltaE = deltaED + (lambda1*deltaEW);
deltaW = -eta1 * deltaE;
dw1 = deltaW;
prevW = prevW + deltaW;

error=deltaE;

for i=2:N1

%prevW=dw1(:,i-1);

	if (error>deltaE)
		eta1= horzcat(eta1,(0.5*eta1(i-1)));
        error=deltaE;
	else
		eta1= horzcat(eta1,eta1(i-1));
	end

	deltaEW=prevW;
	deltaED=-((trainRelArr(i,:) - prevW.'*phi(i,:).')*phi(i,:)).';
	deltaE = deltaED + (lambda1*deltaEW);
	deltaW = -eta1(i) * deltaE;
	
	dw1 = horzcat(dw1,deltaW);
	prevW = prevW + deltaW;
	
end


%%%%%%%%%%%% SGD - Synthetic Data Set %%%%%%%%%%%%
%SGD

w02(1:M2,1) = 5;
%w02= -1+2*rand(M2,1);
%w02= w2+5;
eta2=1;
prevW=w02;

deltaEW=prevW;
deltaED=-((trainRelArrSyn(1,:) - prevW.'*phi3(1,:).')*phi3(1,:)).';
deltaE = deltaED + (lambda2*deltaEW);
deltaW = -eta2 * deltaE;
dw2 = deltaW;
prevW = prevW + deltaW;

error=deltaE;

for i=2:N3
%prevW=dw2(:,i-1);

	if (error>deltaE)
		eta2= horzcat(eta2,(0.5*eta2(i-1)));
        error=deltaE;
	else
		eta2= horzcat(eta2,eta2(i-1));
	end

	deltaEW=prevW;
	deltaED=-((trainRelArrSyn(i,:) - prevW.'*phi3(i,:).')*phi3(i,:)).';
	deltaE = deltaED + (lambda2*deltaEW);
	deltaW = -eta2(i) * deltaE;
	dw2 = horzcat(dw2,deltaW);
	prevW = prevW + deltaW;

end
