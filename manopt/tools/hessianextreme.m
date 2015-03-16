function [y, lambda] = hessianextreme(problem, x, side, y0, options)
% Compute an extreme eigenvector / eigenvalue of the Hessian of a problem.
%
% [u, lambda] = hessianextreme(problem, x, side)
% [u, lambda] = hessianextreme(problem, x, side, u0)
% [u, lambda] = hessianextreme(problem, x, side, u0, options)
% [u, lambda] = hessianextreme(problem, x, side, [], options)
%
% Given a Manopt problem structure and a point x on the manifold problem.M,
% this function computes a tangent vector u at x of unit norm such that the
% Hessian quadratic form is minimized or maximized:
%
%    minimize or maximize <u, Hess f(x)[u]> such that <u, u> = 1,
%
% where <.,.> is the Riemannian metric on the tangent space at x. Choose
% between minimizing and maximizing by setting side = 'min' or 'max', the
% former being the default. The value attained is returned as lambda, and
% is the minimal or maximal eigenvalue of the Hessian (actually, the last
% value attained when the solver stopped). Note that this is a real number
% as the Hessian is a symmetric operator.
%
% The options structure, if provided, will be passed along to manoptsolve.
% As such, you may choose which solver to use to solve the above
% optimization problem by setting options.solver. See manoptsolve's help.
% The other options will be passed along to the chosen solver too.
%
% If u0 is specified, it should be a unit-norm tangent vector at x. It will
% be used as initial guess to solve the above problem.
%
% Often times, it is only necessary to compute a vector u such that the
% quadratic form is negative, if that is at all possible. To do so, you may
% set the following stopping criterion: options.tolcost = -1e-10; (for
% example) and side = 'min'. The solver will return as soon as the
% quadratic cost above drops below the set value.
%
% See also: hessianspectrum manoptsolve

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Aug. 13, 2014.
% Contributors: 
% Change log: 

    % If no initial guess was specified, prepare the empty one.
    if ~exist('y0', 'var')
        y0 = [];
    end

    % If no options are specified, prepare the empty structure.
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    
    % By default, minimize
    if ~exist('side', 'var') || isempty(side)
        side = 'min';
    end
    
    % Convert the side into a sign
    switch lower(side)
        case 'min'
            sign = +1;
        case 'max'
            sign = -1;
        otherwise
            error('The side should be either ''min'' or ''max''.');
    end

    % We define a manifold that is actually the unit sphere on the tangent
    % space to problem.M at x. A generalization would be to consider
    % Stiefel or Grassmann on the tangent space, but this would require
    % manipulating collections of tangent vectors, which in full generality
    % may be more complex (from a programming point of view).
    % Points are represented as tangent vectors of unit norm.
    % Tangent vectors are represented as tangent vectors orthogonal to the
    % root point, with respect to the Riemannian metric on the tangent
    % space.
    
    % M is the original manifold. x is a point on M.
    M = problem.M;
    
    % N is the manifold we build. y will be a point on N, thus also a
    % tangent vector to M at x. This is a typical Riemannian submanifold of
    % a Euclidean space, hence it will be easy to describe in terms of the
    % tools available for M.
    N = tangentspherefactory(M, x);
    
    % Precompute the dbstore here by calling costgrad.
    % An alternative would be to ask for it as an input.
    storedb = struct();
    if canGetGradient(problem)
        [unused1, unused2, storedb] = getCostGrad(problem, x, struct()); %#ok<ASGLU>
    end
    
    % This is the star operator of this party.
    hessian = @(y) getHessian(problem, x, y, storedb);
    
    % Start a Manopt problem structure for the quadratic optimization
    % problem on the sphere N.
    new_problem.M = N;
    
    % Define the cost function, its gradient and its Hessian.

    new_problem.cost = @cost;
    function [f, store] = cost(y, store)
        store = prepare(y, store);
        f = sign*store.f;
    end

    new_problem.grad = @grad;
    function [g, store] = grad(y, store)
        store = prepare(y, store);
        g = N.lincomb(y, sign*2, store.Hy, sign*(-2)*store.f, y);
    end

    new_problem.hess = @hess;
    function [h, store] = hess(y, ydot, store)
        store = prepare(y, store);
        Hydot = hessian(ydot);
        h = N.lincomb(y, sign*2, Hydot, sign*(-2)*store.f, ydot);
        h = N.proj(y, h);
    end

    % This helper makes sure we do not duplicate Hessian computations.
    function store = prepare(y, store)
        if ~isfield(store, 'ready')
            Hy = hessian(y);
            store.f = M.inner(x, y, Hy);
            store.Hy = Hy;
            store.ready = true;
        end
    end
    
    % Call a Manopt solver to solve the quadratic optimization problem on
    % the abstract sphere N.
    [y, lambda] = manoptsolve(new_problem, y0, options);
    lambda = sign*lambda;

end
