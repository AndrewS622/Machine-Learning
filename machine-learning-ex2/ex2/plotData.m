function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
xlabel("Exam 1 score");
ylabel("Exam 2 score");
pos_ind = find(y);
neg_ind = find(~y);
plot(X(pos_ind,1),X(pos_ind,2),'k+', 'LineWidth', 2,'markerSize',7);
hold on
plot(X(neg_ind,1),X(neg_ind,2),'ko', 'MarkerFaceColor', 'y', 'markerSize',7);
legend(['Admitted', 'Not Admitted']);
% =========================================================================



hold off;

end
