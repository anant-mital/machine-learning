% ex 4

% load data
fprintf('Loading data.\n');
%cifar_10_read_data;

% Task 1
tic;
% get random subset of data
RI = unique(randi(10000,1000,1));
fprintf('Classifying subset of %d images.\n', size(RI,1));
T = te_data(RI,:);
TL = te_labels(RI,:);

% get mean colors
fprintf('Extracting features.\n');
ft_te = cifar_10_features(T, 32);
ft_tr = cifar_10_features(tr_data, 32);

% compute normal distribution parameters
fprintf('Getting normal distribution parameters.\n');
[mu, sigma, p] = cifar_10_bayes_learn(ft_tr, tr_labels, false);
[mu, sigmamv, p] = cifar_10_bayes_learn(ft_tr, tr_labels, true);

% classify
fprintf('Classification.\n');
c=cifar_10_bayes_classify(ft_te,mu,sigma,p);
cmv=cifar_10_bayes_classify(ft_te,mu,sigmamv,p);

% evaluate
result = cifar_10_evaluate( c, TL )*100;
resultmv = cifar_10_evaluate( cmv, TL )*100;

fprintf('Classification is %2.2f %% accurate.\n', result);
fprintf('Classification with mnv is %2.2f %% accurate.\n', resultmv);
toc

% Task 2
% get mean colors
fprintf('Task 2.\n');
% N = 16
tic
fte16 = cifar_10_features(T, 16);
ftr16 = cifar_10_features(tr_data, 16);
% compute normal distribution parameters
[mu16, sigma16, p] = cifar_10_bayes_learn(ftr16, tr_labels, true);
% classify
c16=cifar_10_bayes_classify(fte16,mu16,sigma16,p);
% evaluate
result16 = cifar_10_evaluate( c16, TL )*100;
fprintf('Classification is %2.2f %% accurate with N=16.\n', result16);
toc
% N = 8
tic
fte8 = cifar_10_features(T, 8); % te_data);
ftr8 = cifar_10_features(tr_data, 8); % te_data);
[mu8, sigma8, p] = cifar_10_bayes_learn(ftr8, tr_labels, true);
c8=cifar_10_bayes_classify(fte8,mu8,sigma8,p);
result8 = cifar_10_evaluate( c8, TL )*100;
fprintf('Classification is %2.2f %% accurate with N=8.\n', result8);
toc
% N = 4
tic
fte4 = cifar_10_features(T, 4); % te_data);
ftr4 = cifar_10_features(tr_data, 4); % te_data);
[mu4, sigma4, p] = cifar_10_bayes_learn(ftr4, tr_labels, true);
c4=cifar_10_bayes_classify(fte4,mu4,sigma4,p);
result4 = cifar_10_evaluate( c4, TL )*100;
fprintf('Classification is %2.2f %% accurate with N=4.\n', result4);
toc
% N = 2
tic
fte2 = cifar_10_features(T, 2); % te_data);
ftr2 = cifar_10_features(tr_data, 2); % te_data);
[mu2, sigma2, p] = cifar_10_bayes_learn(ftr2, tr_labels, true);
c2=cifar_10_bayes_classify(fte2,mu2,sigma2,p);
result2 = cifar_10_evaluate( c2, TL )*100;
fprintf('Classification is %2.2f %% accurate with N=2.\n', result2);
toc
% N = 1
tic
fte1 = cifar_10_features(T, 1); % te_data);
ftr1 = cifar_10_features(tr_data, 1); % te_data);
[mu1, sigma1, p] = cifar_10_bayes_learn(ftr1, tr_labels, true);
c1=cifar_10_bayes_classify(fte1,mu1,sigma1,p);
result1 = cifar_10_evaluate( c1, TL )*100;
fprintf('Classification is %2.2f %% accurate with N=1.\n', result1);
toc
figure;
plot([32,16,8,4,2,1], [resultmv, result16, result8, result4, result2, result1]);
