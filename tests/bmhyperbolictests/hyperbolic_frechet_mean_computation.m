function [xopt, Y] = hyperbolic_frechet_mean_computation(n, N)
% This is an example of how to use the hyperbolic manifold factory to 
% compute the Frechet mean of points.

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, Nov. 2, 2018
% Contributors:
% Change log:
if ~exist('n', 'var') || isempty(n)
    n = 50;
end
if ~exist('N', 'var') || isempty(N)
    N = 20;
end

% The hyperbolic manifold
problem.M = hyperbolicfactory(n);


% Generate random points on the hyperbolic manifold.
Y = zeros(n+1, N);
for ii = 1 : N
	Y(:, ii) = problem.M.rand();
end

g = ones(n+1,1);
g(1) = -1;

problem.cost = @cost;
problem.grad = @rgrad; % We supply the Riemannian gradient directly.

	function f = cost(x)
		f = 0;
		for ii = 1 : N
			f = f + 0.5*(problem.M.dist(x, Y(:,ii)))^2; % The hyperbolic distance distance square.
		end
		f = f / N;
	end

	function grad = rgrad(x)
		grad = zeros(size(x));
		for ii = 1: N
			grad = grad - problem.M.log(x, Y(:,ii)); % The Riemannian gradient for the squared hyperbolic distance.
		end
		grad = grad / N;
	end

% % Checkgraident
% checkgradient(problem);
% pause;

xopt = conjugategradient(problem);

end