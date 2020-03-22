function f = cifar_10_features(x)
  % convert images to mean colors
  f = zeros(size(x, 1),3);
  f(:,1) = mean(double(x(:,1:1024)), 2);
  f(:,2) = mean(double(x(:,1025:2048)), 2);
  f(:,3) = mean(double(x(:,2048:3072)), 2);
end
