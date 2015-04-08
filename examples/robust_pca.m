function [U, cost] = robustpca(X, d)
% Compute a robust version of PCA (principal component analysis) on data.
% 
% function [U, cost] = robustpca(X, d)
%
% Given a matrix X of size p by n, such that each column represents a
% point in R^p, this computes U: an orthonormal basis of size p by d such
% that the column space of U captures the points X as well as possible.
% More precisely, the function attempts to compute U as the minimizer,
% over the Grassmann manifold (the set of linear subspaces) of:
%
%  f(U) = (1/n) Sum_{i = 1:n} dist(X(:, i), the space spanned by U)
%       = (1/n) Sum_{i = 1:n} || U*U'*X(:, i) - X(:, i) ||
%
% The output cost represents the sum of distances divided by n achieved
% with the returned U. Notice that norms are not squared, for robustness.
%
% In practice, because this function is nonsmooth, we smooth it with a
% pseudo-Huber loss function of parameter epsilon (noted e for short), and
% we iteratively reduce the smoothing parameter:
%
%   f_e(U) = (1/n) Sum_{i = 1:n} l_e(|| U*U'*X(:, i) - X(:, i) ||)
%
%   with l_e(x) = sqrt(x^2 + e^2) - e (for e = 0, this is absolute value).
%
% The intermediate optimization of the smooth cost over the Grassmann
% manifold are performed using the Manopt toolbox.
%
% There are no guarantees that this code will return the optimal U.
% This code is distributed to illustrate one possible way of optimizing
% a nonsmooth cost function over a manifold, using Manopt with smoothing.
% It would be interesting to study whether this algorithm achieves a
% global optimizer with high probability, under some probabilistic model
% on the data and some conditions on the noise.
% For practical use, the constants in the code would need to be tuned.
%
%
% Here is code for a quick test with data in the plane and fitting a line:
% clear all;
% close all;
% clc;
% % Generate some data points aligned on a subspace
% X = rand(2,1)*(1:30) + .02*randn(2, 30).*[(1:30);(1:30)];
% P = randperm(size(X,2));
% % And add some random outliers to the mix
% outliers = 10;
% X(:, P(1:outliers)) = 30*randn(2, outliers);
% % Center the data
% X = X - repmat(mean(X, 2), [1, size(X, 2)]);
% % Compute a robust PCA on the data and plot the results
% U = robustpca(X, 1);
% scatter(X(1,:), X(2,:));
% hold on;
% plot(U(1)*[-1, 1]*100, U(2)*[-1 1]*100, 'r');
% hold off;
% % Compare to a standard PCA
% [Upca,~,~] = svds(X,1);
% hold on;
% plot(Upca(1)*[-1, 1]*100, Upca(2)*[-1 1]*100, 'k');
% hold off;
% xlim(1.1*[min(X(1,:)), max(X(1,:))]);
% ylim(1.1*[min(X(2,:)), max(X(2,:))]);

% This file is part of Manopt and is copyrighted. See the license file.
%
% Main author: Nicolas Boumal and Teng Zhang, May 2, 2014
% Contributors:
%
% Change log:
%   March 4, 2015 (NB):
%       Uses a pseudo-Huber loss rather than a Huber loss: this has the
%       nice advantage of being smooth and simpler to code (no if's).

    [p, n] = size(X);
    manifold = grassmannfactory(p, d);
    problem.M = manifold;
    problem.cost = @robustpca_cost;
    problem.egrad = @robustpca_gradient;
	
	% Do classical PCA for the initial guess.
	% This is just one idea: it is not necessarily useful or ideal.
    [U, ~, ~] = svds(X, d);

	% Iteratively reduce the smoothing constant epsilon and optimize
	% the cost function over Grassmann.
    epsilon = 1;
    
	n_iterations = 6;
	reduction = .5;
	options.verbosity = 2; % Change this number for more or less output
    warning('off', 'manopt:getHessian:approx');
    for iter = 1 : n_iterations
        U = trustregions(problem, U, options);
        epsilon = epsilon * reduction;
    end
    warning('on', 'manopt:getHessian:approx');
    
	% Return the cost as the actual sum of distances, not smoothed.
	epsilon = 0;
	cost = robustpca_cost(U);

    function value = robustpca_cost(U)

        vecs = U*(U'*X) - X;
        sqnrms = sum(vecs.^2, 1);
        vals = sqrt(sqnrms + epsilon^2) - epsilon;
        value = mean(vals);

    end

    function G = robustpca_gradient(U)

		% Note that the computation of vecs and sqnrms is redundant
		% with their computation in the cost function. To speed
		% up the code, it would be wise to use the caching capabilities
		% of Manopt (the store structure). See online documentation.
		% It is not done here to keep the code a bit simpler.
        UtX = U'*X;
        vecs = U*UtX-X;
        sqnrms = sum(vecs.^2, 1);
        % This explicit loop is a bit slow: the code below is equivalent
        % and faster to compute the gradient.
        % G=zeros(p, d);
        % for i=1:n
        %     G = G + (1/sqrt(sqnrms(i) + epsilon^2)) * vecs(:,i) * UtX(:,i)';
        % end
        % G = G/n;
        G = mean(multiscale(1./sqrt(sqnrms + epsilon^2), ...
                           multiprod(reshape(vecs, [p, 1, n]), ...
                              multitransp(reshape(UtX, [d, 1, n])))), 3);
    end

end
