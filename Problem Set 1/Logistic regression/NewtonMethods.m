%% Implement Newton's method for logistic regression

% Load the training data
load q1x.dat;
x1 = q1x(:,1);
x2 = q1x(:,2);
x0 = ones(length(x1),1);
X = [x0,x1,x2];  % Training data

load q1y.dat;
Y = q1y(:,1);   % Training data 

% Implement Newton'method for optimizing maximum likelihood function
n = size(X,2);  % The number of features
theta = zeros(n,1);

iter_time = 50;  % Iteration times of Newton's method
for k = 1:iter_time
    gradient =zeros(n,1);
    H = zeros(n,n);
    for i = 1:length(Y)
       gradient = gradient + (Y(i) - sigmoid(X(i,:)*theta))*X(i,:)';
       H = H - sigmoid(X(i,:)*theta).*(1-sigmoid(X(i,:)*theta)).*X(i,:)'*X(i,:);
    end
    theta = theta - inv(H)*gradient;
end

% Plot training data and the decision boundary fit by logitic regression
% Find the corresponding x when Y =0

figure(1);
plot(X(Y==0,2),X(Y==0,3),'*r');
hold on;
plot(X(Y==1,2),X(Y==1,3),'+b');
hold on;
a = min(X(:,2)):.01:max(X(:,2));
b = (-theta(1)-theta(2)*a)/theta(3);
plot(a,b,'-k');
legend('Y==0','Y==1');
xlabel('x1'),ylabel('x2');
title('The training data');

%The following is the reference code
% figure;hold on;
% for i=1:length(Y)
%     if(Y(i)==0)
%         plot(X(i,2),X(i,3),'rx');
%     else plot(X(i,2),X(i,3),'go');
%     end
% end
% x = min(X(:,2)):.01:max(X(:,2));
% y = -theta(1)/theta(3)-theta(2)/theta(3)*x;
% plot(x,y);




