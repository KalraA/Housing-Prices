function [X_norm, mu, sigma] = featureNormalize(X)

%The normalization formula is (x - mu)/sigma where mu is the mean and sigma is the std deviation. 
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
X_norm = X;
for i = 1:size(X, 2)
mu(i) = mean(X(:, i));
sigma(i) = std(X(:, i));
X_norm(:, i) -= mu(i);
X_norm(:, i) *= 1/sigma(i);
end
end
