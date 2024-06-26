function [X, min_distance] = packing_in_a_ball(d, n, sigma)
% Return a set of points spread out in a ball.
%
% function [X, min_distance] = packing_in_a_ball(d, n, sigma)
%
% Using optimization through a lift in Manopt, this function returns a set
% of n points in R^d with norm <= 1 in the form of a matrix X of size nxd,
% such that the points are spread out in the unit ball.
%
%    Read more about lifts here:  https://www.manopt.org/lifts.html
%
% Ideally, we would maximize the minimum distance between any two points
% X(i, :) and X(j, :), i~=j, but that is a nonsmooth cost function.
% Instead, we smooth that cost function with a classical log-sum-exp
% approximation and (attempt to) solve:
%
%    min_{X in OB(d, n)} log( sum_{i,j} exp(-||xi - xj||^2 / (2sigma^2) ) )
%
% with xi = X(:, i), where sigma > 0 is a smoothing constant. As sigma
% goes to zero, the cost function is a sharper approximation of our target,
% but the cost function becomes stiffer and hence harder to optimize.
%
% The second output, min_distance, is the minimum distance between any two
% points in the returned X. This number is the one we truly are trying to
% maximize.
%
% Notice that this cost function is invariant under rotation of X:
% 
%    f(X) = f(XQ) for all orthogonal Q in O(d).
% 
% We could take the quotient of the oblique manifold OB(d, n) by O(d) to
% remove this symmetry: see elliptopefactory.
%
% See also: elliptopefactory packing_on_the_sphere

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, June 26, 2024
% Contributors:

    if ~exist('d', 'var') || isempty(d)
        % Dimension of the embedding space: R^d
        d = 2;
    end
    if ~exist('n', 'var') || isempty(d)
        % Number n of points to place in the ball in R^d.
        n = 64;
    end
    if ~exist('sigma', 'var') || isempty(sigma)
        % This value should be tuned carefully.
        sigma = 0.05;
    end

    % Choose the Manopt lift that allows us to optimize over a product of n
    % balls in R^d. To pack un a cube, just call cubeslift instead.
    lift = ballslift(d, n);
    
    % Transforms a Gram matrix G to a squared Euclidean distance matrix.
    % Uses cdiag instead of diag to be compatible with AD.
    gram2edm = @(G) cdiag(G)*ones(1, n) + ones(n, 1)*cdiag(G).' - 2*G;
    
    % Cost function in R^(dxn), unconstrained.
    downstairs.cost = @(X) 2*sigma^2*log(sum( ...
                                exp(-gram2edm(X.'*X)/(2*sigma^2)), 'all'));
    
    % Lift the problem to a smooth manifold from where we can smoothly
    % parameterize the product of balls. Also use automatic
    % differentiation (AD) to get the gradient and Hessian of the cost.
    upstairs = manoptlift(downstairs, lift, 'AD');
    
    % We would like to reach a fairly accurate critical point.
    options.tolgradnorm = 1e-9;

    % Run a smooth optimization algorithm on the lifted problem.
    Y = trustregions(upstairs, [], options);

    % Map the computed solution down to the domain downstairs: the columns
    % of X are guaranteed to be in the ball up to machine precision.
    X = lift.phi(Y);

    % Figure out the minimal distance between any two distinct point of X.
    % That is what we actually want to maximize. Minimizing the cost
    % function defined above is merely a proxy for that goal.
    [I, J] = find(triu(ones(n), 1));
    ij = sub2ind([n, n], I, J);
    E = gram2edm(X.'*X);
    min_distance = sqrt(min(E(ij)));

    
    % Some code to display the results
    if d == 2  % if we are working in a disk
        clf;
        plot(X(1, :), X(2, :), '.', 'MarkerSize', 20);
        hold all;
        t = linspace(0, 2*pi, 251);
        plot(cos(t), sin(t), 'k-', 'LineWidth', 2);
        plot(0, 0, 'k.', 'MarkerSize', 10);
        axis equal off;
        set(gcf, 'Color', 'w');
        text(.3, -1, sprintf('Minimum distance: %.4g', min_distance));
    elseif d == 3
        clf;
        plot3(X(1, :), X(2, :), X(3, :), '.', 'MarkerSize', 20);
        hold all;
        plot(0, 0, 'k.', 'MarkerSize', 10);
        axis equal off;
        set(gcf, 'Color', 'w');
        title(sprintf('Minimum distance: %.4g', min_distance));
    end

end
