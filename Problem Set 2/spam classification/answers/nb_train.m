
[spmatrix, tokenlist, trainCategory] = readMatrix('MATRIX.TRAIN');

trainMatrix = full(spmatrix);
numTrainDocs = size(trainMatrix, 1);
numTokens = size(trainMatrix, 2);

% trainMatrix is now a (numTrainDocs x numTokens) matrix.
% Each row represents a unique document (email).
% The j-th column of the row $i$ represents the number of times the j-th
% token appeared in email $i$. 

% tokenlist is a long string containing the list of all tokens (words).
% These tokens are easily known by position in the file TOKENS_LIST

% trainCategory is a (1 x numTrainDocs) vector containing the true 
% classifications for the documents just read in. The i-th entry gives the 
% correct class for the i-th email (which corresponds to the i-th row in 
% the document word matrix).

% Spam documents are indicated as class 1, and non-spam as class 0.
% Note that for the SVM, you would want to convert these to +1 and -1.


index0 = find(trainCategory(:)==0); % The indices of the non-spam emails
index1 = find(trainCategory(:)==1); % The indices of the spam emails


% YOUR CODE HERE
% This code uses multinomial event model for naive Bayes algorithm and
% Laplace smoothing
probNonspam = log(length(index0)/numTrainDocs); % The probability of the non-spam emails
probSpam = log(length(index1)/numTrainDocs);  % The probability of the spam emails

V = numTokens; % The size of dictionary, applied in the Laplace smoothing
nonspamMatrix = trainMatrix(index0,:);
spamMatrix = trainMatrix(index1,:);

nonspamMatrix_totalwords = sum(sum(nonspamMatrix));
spamMatrix_totalwords = sum(sum(spamMatrix));

phiNonspam = zeros(numTokens,1);
phiSpam = zeros(numTokens,1);

for i = 1: numTokens
    phiNonspam(i) = log((sum(nonspamMatrix(:,i))+1)/(nonspamMatrix_totalwords+ V));
    phiSpam(i) = log((sum(spamMatrix(:,i))+1)/(spamMatrix_totalwords+ V));
end


% % Informal sense of how indicative token i is for the SPAM class by looking
% % at log(p(x_j = i| y = 1)/ p(x_j =i| y =0)). Find the five highest
% % positive value on this criteria.
% indiSPAM = zeros(numTokens,1);
%  for j = 1:numTokens
%     indiSPAM(j) = phiSpam(j) - phiNonspam(j);
% end
% [indiSPAM, index] = sort(indiSPAM,'descend'); 
% % The five highest value: httpaddr,spam,unsubscrib,ebai,valet



