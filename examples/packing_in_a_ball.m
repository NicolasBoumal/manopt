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
% For more on packing disks in a disk (d = 2), see
% https://en.wikipedia.org/wiki/Circle_packing_in_a_circle
% https://erich-friedman.github.io/packing/cirincir/
% Replacing ballslift by cubeslift in the code is interesting too:
% https://erich-friedman.github.io/packing/cirinsqu/
%
% See also: elliptopefactory packing_on_the_sphere

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, June 26, 2024
% Contributors:
% Change log: 
% 
%   April 4, 2025 (NB):
%       Factored out plotting code to allow plotting at each iteration.
%       Added code for another cost function in the comments.

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
    
    % Another interesting function: sum of log of 1/squared distances.
    % This leads to rather different behavior.
    % 
    % kk = find(triu(ones(n), 1));
    % triupvals = @(M) M(kk);
    % downstairs.cost = @(X) -mean(log(triupvals(gram2edm(X.'*X))));
    
    % Lift the problem to a smooth manifold from where we can smoothly
    % parameterize the product of balls. Also use automatic
    % differentiation (AD) to get the gradient and Hessian of the cost.
    upstairs = manoptlift(downstairs, lift, 'AD');
    
    % We would like to reach a fairly accurate critical point.
    options.tolgradnorm = 1e-9;

    % Optionally, we may display the configuration at each iteration.
    options.statsfun = @(Y) plot_points(lift.phi(Y));
    % Alternatively, this can also be achieved with the following code.
    % The advantage is that statsfunhelper allows more flexibility in the
    % inputs and in combining this without other statsfuns with records.
    % For example, we might also record the density at each iteration.
    % options.statsfun = statsfunhelper('norecord', @(Y) plot_points(lift.phi(Y)));

    % Run a smooth optimization algorithm on the lifted problem.
    Y = trustregions(upstairs, [], options);

    % Map the computed solution down to the domain downstairs: the columns
    % of X are guaranteed to be in the ball up to machine precision.
    X = lift.phi(Y);

    % Figure out the minimal distance between any two distinct point of X.
    % That is what we actually want to maximize. Minimizing the cost
    % function defined above is merely a proxy for that goal.
    min_distance = mindistance(X);

    % Final plot
    plot_points(X);

end

% Compute the minimal distance between two of the points
function val = mindistance(X)
    n = size(X, 2);
    [I, J] = find(triu(ones(n), 1));
    ij = sub2ind([n, n], I, J);
    gram2edm = @(G) diag(G)*ones(1, n) + ones(n, 1)*diag(G).' - 2*G;
    E = gram2edm(X.'*X);
    val = sqrt(min(E(ij)));
end
    
% Some code to display the results
function plot_points(X)
    [d, n] = size(X);
    min_distance = mindistance(X);
    if d == 2  % if we are working in a 2-D disk
        clf;
        hold on;
        t = linspace(0, 2*pi, 251);
        r = (min_distance/2);
        for i = 1 : n
            fill(X(1, i) + r*cos(t), ...
                 X(2, i) + r*sin(t), [.3, .4, .5]);
        end
        plot((1+r)*cos(t), (1+r)*sin(t), 'k--', 'LineWidth', 2);
        plot(X(1, :), X(2, :), 'b.', 'MarkerSize', 20);
        plot(cos(t), sin(t), 'k-', 'LineWidth', 2);
        plot(0, 0, 'k.', 'MarkerSize', 10);
        axis equal off;
        set(gcf, 'Color', 'w');
        text(.45, -1.1, sprintf('Minimum distance: %.4g', min_distance));
        density = (n*r^2)/(1+r)^2;
        text(.45, -1.2, sprintf('Density: %.4g', density));
        drawnow;
    elseif d == 3 % if we are working in a 3-D ball
        clf;
        plot3(X(1, :), X(2, :), X(3, :), '.', 'MarkerSize', 20);
        hold on;
        plot(0, 0, 'k.', 'MarkerSize', 10);
        axis equal off;
        set(gcf, 'Color', 'w');
        title(sprintf('Minimum distance: %.4g', min_distance));
        drawnow;
    end
end
