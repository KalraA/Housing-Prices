function [theta, J_history] = gradDesc(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1); %this is all the previous costs that we then graph to make sure the learning rate is working
p = size(X, 2); %How many parameters/training example
z = zeros(p, 1); %gradient for each of the parameters
%Gradient Descent Algo
for iter = 1:num_iters

    for j = 1:size(X, 2)
      a = 0;
        for i = 1:m
          a = a + ((theta' * X(i, 1:size(X, 2))') - y(i))*X(i, j)/m;
        end
        
        z(j) = a;
    end

    for j = 1:p
        theta(j) -= alpha*z(j);

        J_history(iter) = cost(X, y, theta);

    end

end
