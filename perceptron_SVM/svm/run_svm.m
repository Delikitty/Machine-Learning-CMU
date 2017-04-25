function [model] = run_svm(X,Y,C)

%% train SVM
Y = double(Y);
Y(Y==0) = -1;
model =fitcsvm(X, Y, 'BoxConstraint', C);
sv = model.SupportVectors;

%% plot the data points
figure()
title(sprintf('SVM Linear Classifier'), 'FontSize', 14)
subplot(1,2,1);
gscatter(X(:,1), X(:,2), Y, 'cr');
hold on;
plot(sv(:,1), sv(:,2), 'ko', 'Markersize', 10)
legend('Negative', 'Positive', 'Support vector');

%% plot the decision boundary
d = 0.02; % Step size of the grid
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];        % The grid
[~,scores1] = predict(model,xGrid); % The scores
contour(x1Grid,x2Grid,reshape(scores1(:,2),size(x1Grid)),[0 0],'k-');
contour(x1Grid,x2Grid,reshape(scores1(:,2),size(x1Grid)),[1 1],'r--');
contour(x1Grid,x2Grid,reshape(scores1(:,2),size(x1Grid)),[-1 -1],'c--');

%% compute slack variables and plot points based on the slack values
subplot(1,2,2);
beta= model.Beta;
bias = model.Bias;
TOL = 0.01;
e = double(Y) .* (X * beta + bias); % slack variables
K = zeros(length(e),1);
K(e<1-TOL) = 1;
K(e>1+TOL) = 2;
h = gscatter(X(:,1), X(:,2), K, 'crk','', '', 'doleg');
hold on;

%% plot decision boundary and margin
d = 0.02; % Step size of the grid
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];        % The grid
[~,scores1] = predict(model,xGrid); % The scores
contour(x1Grid,x2Grid,reshape(scores1(:,2),size(x1Grid)),[0 0],'k-');
h1_ = plot(NaN, 'k-');
contour(x1Grid,x2Grid,reshape(scores1(:,2),size(x1Grid)),[1 1],'r--');
h2_ = plot(NaN, 'c--');
contour(x1Grid,x2Grid,reshape(scores1(:,2),size(x1Grid)),[-1 -1],'c--');
h3_ = plot(NaN, 'r--');
legend([h1_, h2_, h3_], 'Decision Boundary', 'Negative Hyperplane', 'Positive Hyperplane')
% suptitle('SVM Linear Classifier');
end
