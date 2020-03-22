function [mu,sigma,p]=cifar_10_bayes_learn(F,labels)
  % uL = unique(labels)';
  % Go through each label
  for L = min(labels):max(labels)
      I = find(labels==L);
      % Mean values
      mu(L+1,:) = [mean(F(I,1)), mean(F(I,2)), mean(F(I,3))];
      
      % Standard deviance 
      % sigma(L+1,:) = [ ...
      %     sum((F(I,1)-mu(L+1,1)).^2), ...
      %     sum((F(I,2)-mu(L+1,2)).^2), ...
      %     sum((F(I,3)-mu(L+1,3)).^2)];
      % sigma(L+1,:) = sigma(L+1,:)./(size(F(I), 1)-1);
      
      % built in matlab command does the same
      sigma(L+1,:) = [std(F(I,1)),std(F(I,2)), std(F(I,3))];
      
      % expected value
      % p(L+1,1:3) = double(size(I, 1))/double(size(labels, 1));
      p(L+1) = double(size(I, 1))/double(size(labels, 1));
  end
  % need to take sqrt if not using built in matlab command
  % sigma = sqrt(sigma); 
end
