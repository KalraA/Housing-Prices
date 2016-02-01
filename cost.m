function J = cost(X, y, theta)

m = length(y); % number of training examples
a = 0;
for i = 1:m
  a = a + (theta' * X(i, 1:size(X, 2))' - y(i, 1))^2;
end

J = a/(2*m);

% J = 1/2m * (squared error sum of each training example)

end
