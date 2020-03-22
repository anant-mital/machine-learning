% load data
fprintf('Loading data.\n');
cifar_10_read_data;

tic;
% get random subset of data
RI = unique(randi(10000,1000,1));
fprintf('Classifying subset of %d images.\n', size(RI,1));
T = te_data; % (RI,:);
TL = te_labels; % (RI,:);

% get mean colors
fprintf('Extracting features.\n');
fte = cifar_10_features(T); % te_data);
ftr = cifar_10_features(tr_data); % te_data);

% compute normal distribution parameters
fprintf('Getting normal distribution parameters.\n');
[mu, sigma, p] = cifar_10_bayes_learn(ftr, tr_labels);

% classify
fprintf('Classification.\n');
c=cifar_10_bayes_classify(fte,mu,sigma,p);

% evaluate
evaluation_accuracy = cifar_10_evaluate( c, TL )*100;
fprintf('Classification is %2.2f %% accurate.\n', evaluation_accuracy);
toc

% confusion matrix
ConfMat = zeros(length(unique(TL)));
for pred = 0:9
  IC = find(c==pred);
  for real = 0:9
    ConfMat(real+1, pred+1) = length(find(TL(IC)==real));
  end
end

% visualize confusion matrix
figure;
imagesc(ConfMat);
colormap gray;

%plot nonmixing
p1 = 1;
p2 = 7;
x=80:160;
figure;
subplot(3,1,1);
plot(x,normpdf(x,mu(p1, 1)),'r-',x,normpdf(x,mu(p2,1)),'r--')
subplot(3,1,2);
plot(x,normpdf(x,mu(p1, 2)),'g-',x,normpdf(x,mu(p2,2)),'g--')
subplot(3,1,3);
plot(x,normpdf(x,mu(p1, 3)),'b-',x,normpdf(x,mu(p2,3)),'b--')
legend(label_names{p1}, label_names{p2});

%plot mixing
p1 = 4;
p2 = 7;
x=80:160;
figure;
subplot(3,1,1);
plot(x,normpdf(x,mu(p1, 1)),'r-',x,normpdf(x,mu(p2,1)),'r--')
subplot(3,1,2);
plot(x,normpdf(x,mu(p1, 2)),'g-',x,normpdf(x,mu(p2,2)),'g--')
subplot(3,1,3);
plot(x,normpdf(x,mu(p1, 3)),'b-',x,normpdf(x,mu(p2,3)),'b--')
legend(label_names{p1}, label_names{p2});
