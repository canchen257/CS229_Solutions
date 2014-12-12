%% Implement (unweighted) linear regression on the dataset(Using the normal equations)
% Load the training dataset
load q2x.dat;
x1 = q2x(:,1);
x0 = ones(length(x1),1); % Include the intercept term
X = [x0, x1];
load q2y.dat;
Y = q2y(:,1);

% Using normal equation to implement unweighted linear regression
% The normal equation Theta = inv(X'X)*X'Y
theta = zeros(size(X,2),1);
theta = inv(X'*X)*X'*Y;

% Plot the dataset and the straight line
figure; hold on;
plot(x1,Y,'ro');
a = min(X(:,2)):0.01:max(X(:,2));
b = theta(1) + theta(2) * a;
plot(a,b);
xlabel('x');ylabel('y');
title('The unweighted linear regression');