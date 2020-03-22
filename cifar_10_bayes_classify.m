function c=cifar_10_bayes_classify(f,mu,sigma,p)
  P = []; 
  % go through each class
  for CN = 1:length(p) 
    % compute normal distributions for each color feature
    NR = normpdf(f(:,1),mu(CN,1),sigma(CN,1));
    NG = normpdf(f(:,2),mu(CN,2),sigma(CN,2));
    NB = normpdf(f(:,3),mu(CN,3),sigma(CN,3));
    % Get bayesian probability
    P(:,CN) = NR.*NG.*NB*p(CN);
  end
  % find the most probable class/label for each image
  [~, I] = max(P, [], 2);
  c = I-1;
end
