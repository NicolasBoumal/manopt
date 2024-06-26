function [X, maxdot] = packing_on_the_sphere(d, n, epsilon, X0)
% Return a set of points spread out on the sphere.
%
% function [X, maxdot] = packing_on_the_sphere(d, n, epsilon, X0)
%
% Using optimization on the oblique manifold, that is, the product of
% spheres, this function returns a set of n points with unit norm in R^d in
% the form of a matrix X of size nxd, such that the points are spread out
% on the sphere.
%
% Ideally, we would minimize the maximum inner product between any two
% points X(i, :) and X(j, :), i~=j, but that is a nonsmooth cost function.
% Instead, we replace the max function by a classical log-sum-exp
% approximation and (attempt to) solve:
%
%    min_{X in OB(d, n)} log( .5*sum_{i~=j} exp( (xi'*xj)/epsilon ) )
%
% with xi = X(:, i), where epsilon is some "diffusion constant". As epsilon
% goes to zero, the cost function is a sharper approximation of the max
% function (under some assumptions), but the cost function becomes stiffer
% and hence harder to optimize.
%
% The second output, maxdot, is the maximum inner product between any two
% points in the returned X. This number is the one we truly are trying to
% minimize.
%
% Notice that this cost function is invariant under rotation of X:
% 
%    f(X) = f(XQ) for all orthogonal Q in O(d).
% 
% We could take the quotient of the oblique manifold OB(d, n) by O(d) to
% remove this symmetry: see elliptopefactory.
%
% This is known as the Thomson or, more specifically, the Tammes problem:
%
%    http://en.wikipedia.org/wiki/Tammes_problem
% 
% Neil Sloane collects the best known packings here:
% 
%    http://neilsloane.com/packings/
%
% See also: elliptopefactory packing_in_a_ball

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, July 2, 2013
% Contributors:
%
% Change log:
%   Aug. 14, 2013 (NB) : Code now compatible to experiment with both the
%                        obliquefactory and the elliptopefactory.
%
%   Jan.  7, 2014 (NB) : Added reference to Neil Sloane's page and the
%                        maxdot output.
%
%   June 24, 2014 (NB) : Now shifting exponentials to alleviate numerical
%                        trouble when epsilon is too small.
%   
%   Aug. 31, 2021 (XJ) : Added AD to compute the gradient
%   
%   June 26, 2024 (NB) : Modernized the code and comments somewhat.

    if ~exist('d', 'var') || isempty(d)
        % Dimension of the embedding space: R^d
        d = 3;
    end
    if ~exist('n', 'var') || isempty(n)
        % Number n of points to place of the sphere in R^d.
        % For example, n=12 should yield an icosahedron:
        %    https://en.wikipedia.org/wiki/Icosahedron
        % Notice though that platonic solids are not always optimal.
        % Try for example n = 8: you don't get a cube.
        n = 24;
    end
    if ~exist('epsilon', 'var') || isempty(epsilon)
        % This value should be as close to 0 as affordable.
        % If it is too close to zero, optimization first becomes much
        % slower, than simply doesn't work anymore becomes of floating
        % point overflow errors (NaN's and Inf's start to appear).
        % If it is too large, then log-sum-exp is a poor approximation of
        % the max function, and the spread will be less uniform.
        % An okay value seems to be 0.01 or 0.001 for example. Note that a
        % better strategy than using a small epsilon straightaway is to
        % reduce epsilon bit by bit and to warm-start subsequent
        % optimization in that way. Trustregions will be more appropriate
        % for these fine tunings.
        epsilon = 0.0015;
    end
    
    % Pick your manifold.
    % By default we work with the oblique manifold (n rows each of norm 1).
    % The elliptope factory quotients out the global rotation invariance of
    % the problem, which is more natural but conceptually more complicated.
    % For usage with the toolbox it is the same though.
    %
    manifold = obliquefactory(n, d, 'rows');
    % manifold = elliptopefactory(n, d);
    
    % Generate a random initial guess if none was given.
    if ~exist('X0', 'var') || isempty(X0)
        X0 = manifold.rand();
    end

    % Define the cost function and its derivatives, with caching: the store
    % structure we receive as input is tied to the input point X. Everytime
    % this cost function is called at /this/ point X, we receive the /same/
    % store structure back. We may modify the store structure inside
    % the function and return it: the changes are remembered for next time.
    function store = prepare(X, store)
        if ~isfield(store, 'ready')
            XXt = X*X';
            % Shift the exponentials by the maximum value to reduce
            % numerical trouble due to possible overflows.
            s = max(max(triu(XXt, 1)));
            expXXt = exp((XXt-s)/epsilon);
            % Zero out the diagonal
            expXXt(1:(n+1):end) = 0;
            u = sum(sum(triu(expXXt, 1)));
            store.s = s;
            store.expXXt = expXXt;
            store.u = u;
            store.ready = true;
        end
    end
    function [f, store] = cost(X, store)
        store = prepare(X, store);
        u = store.u;
        s = store.s;
        f = s + epsilon*log(u);
    end

    % Define the Euclidean gradient of the cost.
    function [G, store] = egrad(X, store)
        store = prepare(X, store);
        % Compute the Euclidean gradient
        G = store.expXXt*X / store.u;
    end

    % Setup the problem structure with its manifold M and cost + egrad
    % functions.
    problem.M = manifold;
    problem.cost = @cost;
    problem.egrad = @egrad;

    % An alternative way to compute the grad is to use automatic
    % differentiation provided in the deep learning toolbox (slower).
    % Notice that the function triu is not supported for AD so far.
    % Replace it with ctriu described in the file manoptADhelp.m
    % problem.cost = @cost_AD;
    % function f = cost_AD(X)
    %    XXt = X*X';
    %    s = max(max(ctriu(XXt, 1)));
    %    expXXt = exp((XXt-s)/epsilon);
    %    expXXt(1:(n+1):end) = 0;
    %    u = sum(sum(ctriu(expXXt, 1)));
    %    f = s + epsilon*log(u);
    % end
    % Call manoptAD to prepare AD for the problem structure
    % problem = manoptAD(problem);
    
    % For debugging, it's always nice to check the gradient a few times.
    % checkgradient(problem);
    % pause;
    
    % Call a solver on our problem with a few options defined. We did not
    % specify the Hessian but it is still okay to call trustregions: Manopt
    % approximates the Hessian with finite differences of the gradient.
    opts.tolgradnorm = 1e-8;
    opts.maxtime = 1200;
    opts.maxiter = 1e5;
    % X = trustregions(problem, X0, opts);
    X = conjugategradient(problem, X0, opts);
    
    % Evaluate the maximum inner product between any two points of X.
    XXt = X*X';
    dots = XXt(find(triu(ones(n), 1))); %#ok<FNDSB>
    maxdot = max(dots);
    
    % Similarly, even though we did not specify the Hessian, we may still
    % estimate its spectrum at the solution. It should reflect the
    % invariance of the cost function under a global rotation of the
    % sphere, which is an invariance under the group O(d) of dimension
    % d(d-1)/2 : this translates into d(d-1)/2 zero eigenvalues in the
    % spectrum of the Hessian.
    % The approximate Hessian is not a linear operator, and it is a
    % fortiori not symmetric. The result of this computation is thus not
    % precise. It does display the zero eigenvalues as expected though.
    if manifold.dim() < 300
        evs = hessianspectrum(problem, X);
        figure;
        stem(1:length(evs), sort(evs), '.');
        title(['Eigenvalues of the approximate Hessian of the cost ' ...
               'function at the solution']);
    end
    
    
    % Show how the distances between points are distributed.
    figure;
    histogram(real(acos(dots)), 20);
    title('Histogram of the geodesic distances');
    
    % This is the quantity we actually want to minimize.
    fprintf('Maximum inner product between two points: %g\n', maxdot);
    
    
    % Give some visualization if the dimension allows it.
    if d == 2
        % For the circle, the optimal solution consists in spreading the
        % points with angles uniformly sampled in (0, 2pi). This
        % corresponds to the following value for the max inner product:
        fprintf('Optimal value for the max inner product: %g\n', ...
                cos(2*pi/n));
        figure;
        t = linspace(-pi, pi, 201);
        plot(cos(t), sin(t), '-', ...
             'LineWidth', 3, 'Color', [152, 186, 220]/255);
        daspect([1, 1, 1]);
        box off;
        axis off;
        hold on;
        plot(X(:, 1), X(:, 2), 'r.', 'MarkerSize', 25);
        hold off;
    end
    if d == 3
        figure;
        set(gcf, 'Color', 'w');
        % Plot the sphere
        [sphere_x, sphere_y, sphere_z] = sphere(50);
        handle = surf(sphere_x, sphere_y, sphere_z);
        set(handle, 'FaceColor', [152, 186, 220]/255);
        set(handle, 'FaceAlpha', .5);
        set(handle, 'EdgeColor', [152, 186, 220]/255);
        set(handle, 'EdgeAlpha', .5);
        daspect([1, 1, 1]);
        set(gca, 'Clipping', 'off');
        box off;
        axis off;
        hold on;
        % Add the chosen points
        Y = 1.01*X';
        plot3(Y(1, :), Y(2, :), Y(3, :), 'r.', 'MarkerSize', 25);
        % And connect the points which are at minimal distance,
        % within some tolerance.
        min_distance = real(acos(maxdot));
        for i = 1:n
            for j = (i+1):n
                yi = Y(:, i);
                yj = Y(:, j);
                if real(acos(yi'*yj)) <= 1.20*min_distance
                    plot3([yi(1), yj(1)], [yi(2), yj(2)], ...
                                          [yi(3), yj(3)], 'k-');
                end
            end
        end
        hold off;
    end

end
