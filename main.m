%% Getting data
data = load('housingData.txt');
X = data(:, 1:2); %The first two colums are the sqr feet and the bedrooms
y = data(:, 3); %The last column is the price
m = length(y); %m is the total number of training examples

% Scale features and set them to zero mean
size(X)
[X mu sigma] = featureNormalize(X); %Running the feature normalization
size(X)
% Add bias term to X for the Theta
X = [ones(m, 1) X];

alpha = 0.01; %Picking a random learning rate
num_iters = 471; %Picking a random number of iterations

%Initializing theta to a zero matrix as a starting point
theta = zeros(3, 1);
[theta, J_history] = gradDesc(X, y, theta, alpha, num_iters);

% plot cost as iterations increase to help detect effectiveness of the learning rate.
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display Theta after it has been computed.
fprintf('Theta is: \n');
fprintf(' %f \n', theta);


% Estimate the price of a 1850 sq-ft, 3 br house
price = theta(1)*1 + theta(2)*(1850 - mu(1))/sigma(1) + theta(3)*(3 - mu(2))/sigma(2); % You should change this

fprintf(['Predicted Price w/ GD:\n $%f\n'], price);

% Normal Equation Time!
% Reload data because we feature normalized for GD, but we dont do that for NE
data = load('housingData.txt');
X = data(:, 1:2); %The first two colums are the sqr feet and the bedrooms
y = data(:, 3); %The last column is the price
m = length(y); %m is the total number of training examples

% Add bias term to X for the Theta
X = [ones(m, 1) X];

% Use normal equation to calcuate Theta
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta is: \n');
fprintf(' %f \n', theta);


% Estimate the price of a 1850 sq-ft, 3 br house
price = theta(1)*1 + theta(2)*1850 + theta(3)*3; % You should change this

fprintf(['Predicted Price w/ NE:\n $%f\n'], price);
