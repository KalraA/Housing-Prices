function [theta] = normalEqn(X, y)
theta = pinv(X'*X)*X'*y; %The normal equation
end
