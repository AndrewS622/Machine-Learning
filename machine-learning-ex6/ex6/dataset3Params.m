function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% make vectors and matrices to test parameter combinations
params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
n = length(params);
res = zeros(n);

% loop through all parameter combinations, fit model, and store error
for i = 1:n
    for j = 1:n
        C = params(i);
        sigma = params(j);
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        res(i, j) = mean(double(predictions ~= yval));
    end
end

% find parameter combination which yielded smallest error
[~, is] = min(res);
[~, j] = min(min(res));
i = is(j);
C = params(i);
sigma = params(j);
% =========================================================================

end
