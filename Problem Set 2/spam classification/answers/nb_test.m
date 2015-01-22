
[spmatrix, tokenlist, category] = readMatrix('MATRIX.TEST');

testMatrix = full(spmatrix);
numTestDocs = size(testMatrix, 1);
numTokens = size(testMatrix, 2);

% Assume nb_train.m has just been executed, and all the parameters computed/needed
% by your classifier are in memory through that execution. You can also assume 
% that the columns in the test set are arranged in exactly the same way as for the
% training set (i.e., the j-th column represents the same token in the test data 
% matrix as in the original training data matrix).

% Write code below to classify each document in the test set (ie, each row
% in the current document word matrix) as 1 for SPAM and 0 for NON-SPAM.

% Construct the (numTestDocs x 1) vector 'output' such that the i-th entry 
% of this vector is the predicted class (1/0) for the i-th  email (i-th row 
% in testMatrix) in the test set.
output = zeros(numTestDocs, 1);

%---------------
% YOUR CODE HERE
testNonspam = zeros(numTestDocs,1); % calculate the logorithm of prob. of Nonspam
testSpam = zeros(numTestDocs,1); % calculate the logorithm of prob. of Spam

for k = 1: numTestDocs
    testNonspam(k) = sum(phiNonspam(find(testMatrix(k,:) ~=0))) + probNonspam;
    testSpam(k) = sum(phiSpam(find(testMatrix(k,:) ~=0))) + probSpam;
%     output(k) = (testSpam(k) > testNonspam(k));
    if(testNonspam(k) > testSpam(k))
        output(k) = 0;
    else output(k) = 1;
    end
end
%---------------

% Compute the error on the test set
error=0;
for i=1:numTestDocs
  if (category(i) ~= output(i))
    error=error+1;
  end
end

%Print out the classification error on the test set
error/numTestDocs


