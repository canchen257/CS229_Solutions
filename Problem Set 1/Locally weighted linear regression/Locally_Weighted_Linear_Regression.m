%% Implement locally weighted linear regression on the dateset
%  Using the generalized weighted normal equation as below
%   Theta = inv(X'WX)X'WY, where W is diagonal matrix

load q2x.dat;
x1 = q2x(:,1);
m = length(x1);
x0 = ones(m,1); % Include the intercept term
X = [x0, x1];
load q2y.dat;
Y = q2y(:,1);

% Using the generalized weighted normal equation to implement unweighted linear regression
% The normal equation Theta = inv(X'WX)X'WY
theta = zeros(size(X,2),1);
tau = 0.8; % The bandwidth parameter
reg_x = min(X(:,2)):0.1:max(X(:,2));
reg_y = zeros(size(reg_x,2),1);
for k = 1:size(reg_x,2)
    W = zeros(m,1);
    for i = 1:m
        W(i) = exp((-(reg_x(k)-x1(i))^2)/2*tau^2);
    end
    W = diag(W); % Transfer to diagnal matrix
    theta = pinv(X'*W*X)*X'*W*Y;
    reg_y(k) = theta(1) + theta(2)*reg_x(k);
end


% Plot the dataset and the straight line
figure; hold on;
plot(x1,Y,'ro');
plot(reg_x,reg_y);
xlabel('x');ylabel('y');
title('The logically weighted linear regression');