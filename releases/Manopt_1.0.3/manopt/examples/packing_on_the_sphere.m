function X = packing_on_the_sphere(d, n, epsilon, X0)
% Return a set of points spread out on the sphere.
%
% function X = packing_on_the_sphere(d, n, epsilon, X0)
%
% Using optimization on the oblique manifold, that is, the product of
% spheres, this function returns a set of n points with unit norm in R^d in
% the form of a matrix of size dxn, such that the points are spread out on
% the sphere. Ideally, we would minimize the maximum inner product between
% any two points X(:, i) and X(:, j), i~=j, but that is a nonsmooth cost
% function. Instead, we replace the max function by a classical log-sum-exp
% approximation and (attempt to) solve:
%
% min_{X in OB(d, n)} log( .5*sum_{i~=j} exp( xi'*xj/epsilon ) ),
%
% with xi = X(:, i) and epsilon is some "diffusion constant". As epsilon
% goes to zero, the cost function is a sharper approximation of the max
% function (under some assumptions), but the cost function becomes stiffer
% and hencee harder to optimize.
%
% Notice that this cost function is invariant under rotation of X:
% f(X) = f(QX) for all orthogonal Q in O(d).
% This calls for optimization over the set of symmetric positive
% semidefinite matrices of size n and rank d with unit diagonal, which can
% be thought of as the quotient of the oblique manifold OB(d, n) by O(d).
%
% This is known as the Thomson or, more specifically, the Tammes problem:
% http://en.wikipedia.org/wiki/Tammes_problem

% This file is part of Manopt and is copyrighted. See the license file.
%
% Main author: Nicolas Boumal, July 2, 2013
% Contributors:
%
% Change log:
%   
    clc; close all;

    import manopt.tools.*;
    import manopt.manifolds.oblique.*;
    import manopt.solvers.trustregions.*;
    import manopt.solvers.conjugategradient.*;
    
    if ~exist('d', 'var') || isempty(d)
        % Dimension of the embedding space: R^d
        d = 3;
    end
    if ~exist('n', 'var') || isempty(n)
        % Number n of points to place of the sphere in R^d.
        % For example, n=12 yields an icosahedron:
        % https://en.wikipedia.org/wiki/Icosahedron
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
        % An okay value seems to be 0.01 or 0.001 for example.
        epsilon = 0.0015;
    end
    
    % Pick your manifold.
    OB = obliquefactory(d, n);
    
    % Generate a random initial guess if none was given.
    if ~exist('X0', 'var') || isempty(X0)
        X0 = OB.rand();
    end

    % Define the cost function with caching system used: the store
    % structure we receive as input is tied to the input point X. Everytime
    % this cost function is called at this point X, we will receive the
    % same store structure back. We may modify the store structure inside
    % the function and return it: the changes will be remembered for next
    % time.
    function [f store] = cost(X, store)
        if ~isfield(store, 'ready')
            XtX = X'*X;
            expXtX = exp(XtX/epsilon);
            % Zero out the diagonal
            expXtX(1:(n+1):end) = 0;
            u = sum(sum(triu(expXtX, 1)));
            store.XtX = XtX;
            store.expXtX = expXtX;
            store.u = u;
            store.ready = true;
        end
        u = store.u;
        f = epsilon*log(u);
    end

    % Define the gradient of the cost. When the gradient is called at a
    % point X for which the cost was already called, the store structure we
    % receive remember everything that the cost function stored in it, so
    % we can reuse previously computed elements.
    function [g store] = grad(X, store)
        if ~isfield(store, 'ready')
            [~, store] = cost(X, store);
        end
        % Compute the Euclidean gradient
        eg = X*store.expXtX / store.u;
        % Convert to the Riemannian gradient (by projection)
        g = OB.egrad2rgrad(X, eg);
    end

    % Setup the problem structure with its manifold M and cost+grad
    % functions.
    problem.M = OB;
    problem.cost = @cost;
    problem.grad = @grad;

    % For debugging, it's always nice to check the gradient a few times.
    % checkgradient(problem);
    % pause;
    
    % Call a solver on our problem with a few options defined. We did not
    % specify the Hessian but it is still okay to call trustregion: Manopt
    % will approximate the Hessian with finite differences of the gradient.
    opts.tolgradnorm = 1e-9;
    opts.maxtime = 1200;
    opts.maxiter = 1e5;
    X = trustregions(problem, X0, opts);
    % X = conjugategradient(problem, X0, opts);
    
    
    % Similarly, even though we did not specify the Hessian, we may still
    % estimate its spectrum at the solution. It should reflect the
    % invariance of the cost function under a global rotatioon of the
    % sphere, which is an invariance under the group O(d) of dimension
    % d(d-1)/2 : this translates into d(d-1)/2 zero eigenvalues in the
    % spectrum of the Hessian.
    % The approximate Hessian is not a linear operator, and is it a
    % fortiori not symmetric. The result of this computation is thus not
    % reliable. It does display the zero eigenvalues as expected though.
    if OB.dim() < 300
        evs = real(hessianspectrum(problem, X));
        figure;
        stairs(sort(evs));
        title(['Eigenvalues of the approximate Hessian of the cost ' ...
               'function at the solution']);
    end
    
    
    % Show how the inner products X(:, i)'*X(:, j) are distributed.
    figure;
    XtX = X'*X;
    x = XtX(find(triu(ones(n), 1))); %#ok<FNDSB>
    hist(real(acos(x)), 20);
    title('Histogram of the geodesic distances');
    
    % This is the quantity we actually want to minimize.
    fprintf('Maximum inner product between two points: %g\n', max(x));
    
    
    % Give some visualization if the dimension allows
    if d == 2
        % For the circle, the optimal solution consists in spreading the
        % points with angles uniformly sampled in (0, 2pi). This
        % corresponds to the following value for the max inner product:
        fprintf('Optimal value for the max inner product: %g\n', cos(2*pi/n));
        figure;
        t = linspace(-pi, pi, 201);
        plot(cos(t), sin(t), '-', 'LineWidth', 3, 'Color', [152,186,220]/255);
        daspect([1 1 1]);
        box off;
        axis off;
        hold on;
        plot(X(1, :), X(2, :), 'r.', 'MarkerSize', 25);
        hold off;
    end
    if d == 3
        figure;
        % Plot the sphere
        [sphere_x sphere_y sphere_z] = sphere(50);
        handle = surf(sphere_x, sphere_y, sphere_z);
        set(handle, 'FaceColor', [152,186,220]/255);
        set(handle, 'FaceAlpha', .5);
        set(handle, 'EdgeColor', [152,186,220]/255);
        set(handle, 'EdgeAlpha', .5);
        daspect([1 1 1]);
        box off;
        axis off;
        hold on;
        % Add the chosen points
        Y = 1.02*X;
        plot3(Y(1, :), Y(2, :), Y(3, :), 'r.', 'MarkerSize', 25);
        % And connect the points which are at minimal distance,
        % within some tolerance.
        max_inner = max(x);
        min_distance = real(acos(max_inner));
        connected = real(acos(XtX)) <= 1.20*min_distance;
        [Ic Jc] = find(triu(connected, 1));
        for k = 1 : length(Ic)
            i = Ic(k); j = Jc(k);
            plot3(Y(1, [i j]), Y(2, [i j]), Y(3, [i j]), 'k-');
        end
        hold off;
    end

end
