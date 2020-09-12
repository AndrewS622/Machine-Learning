function [error_train, error_val] = ...
    learningCurveRandom(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve with multiple random selections
%   [error_train, error_val] = ...
%       LEARNINGCURVERANDOM(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)). Here, n_trial = 50
%       random permutations are sampled for each size in the range [1,m],
%       where m is the number of samples in the training set. Samples are
%       taken both from the training and validation sets of the same size,
%       and results are averaged for each sample size over all trials.
%

% Number of training examples
m = size(X, 1);
m_val = size(Xval, 1);
n_trial = 50;

error_train = zeros(m, 1);
error_val   = zeros(m, 1);
error_tr = zeros(n_trial, 1);
error_v = zeros(n_trial, 1);

for i = 1:m
    for j = 1:n_trial
        perm = randperm(m, i);
        perm_val = randperm(m_val, i);
        theta = trainLinearReg(X(perm,:), y(perm), lambda);
        h = X(perm,:) * theta;
        error_tr(j) = (h - y(perm))' * (h - y(perm)) / (2*i);
        h_val = Xval(perm_val,:) * theta;
        error_v(j) = (h_val - yval(perm_val))' * (h_val - yval(perm_val)) / (2 * i);
    end
    error_train(i) = mean(error_tr);
    error_val(i) = mean(error_v);
end


end
