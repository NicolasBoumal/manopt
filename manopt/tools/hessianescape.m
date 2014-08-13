function [y, lambda] = hessianescape(problem, x, y0, options)
% Compute an extreme eigenvector / eigenvalue of the Hessian of a problem.
%
% [u, lambda] = hessianescape(problem, x)
% [u, lambda] = hessianescape(problem, x, u0)
% [u, lambda] = hessianescape(problem, x, u0, options)
% [u, lambda] = hessianescape(problem, x, [], options)
%
% Given a Manopt problem structure and a point x on the manifold problem.M,
% this function computes a tangent vector u at x of unit norm such that the
% Hessian quadratic form is minimized:
%
%    minimize <u, Hess f(x)[u]> such that <u, u> = 1,
%
% where <.,.> is the Riemannian metric on the tangent space at x. The value
% attained is returned as lambda, and is the minimal eigenvalue of the
% Hessian (actually, the minimal value attained when the sovler stopped).
% Note that this is a real number as the Hessian is a symmetric operator.
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
% example). The solver will return as soon as the quadratic cost above
% drops below the set value.
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
    N = struct();
    
    % u, u1 and u2 will be tangent vectors to N at y. The tangent space to
    % N at y is a subspace of the tangent space to M at x, thus u, u1 and
    % u2 are also tangent vectors to M at x.
    
    N.dim   = @() M.dim() - 1;
    N.inner = @(y, u1, u2) M.inner(x, u1, u2);
    N.norm  = @(y, u)      M.norm(y, u);
    N.proj  = @(y, v) M.lincomb(x, 1, v, -M.inner(x, v, y), y);
    N.typicaldist = @() 1;
    N.tangent = N.proj;
    N.egrad2rgrad = N.proj;
    N.retr = @retraction;
    N.exp = N.retr;
    function yy = retraction(y, u, t)
        if nargin == 2
            t = 1;
        end
        y_plus_tu = M.lincomb(x, 1, y, t, u);
        nrm = M.norm(x, y_plus_tu);
        yy = M.lincomb(x, 1/nrm, y_plus_tu);
    end
    N.rand = @random;
    function y = random()
        y = M.randvec(x);
        nrm = M.norm(x, y);
        y = M.lincomb(x, 1/nrm, y);
    end
    N.randvec = @randvec;
    function u = randvec(y)
        u = N.proj(y, N.rand());
        nrm = N.norm(y, u);
        u = M.lincomb(x, 1/nrm, u);
    end
    N.zerovec = M.zerovec;
    N.lincomb = M.lincomb;
    N.transp = @(y1, y2, u) N.proj(y2, u);
    N.hash = @(y) ['z' hashmd5(M.vec(x, y))];
    
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
        f = store.f;
    end

    new_problem.grad = @grad;
    function [g, store] = grad(y, store)
        store = prepare(y, store);
        g = N.lincomb(y, 2, store.Hy, -2*store.f, y);
    end

    new_problem.hess = @hess;
    function [h, store] = hess(y, ydot, store)
        store = prepare(y, store);
        Hydot = hessian(ydot);
        h = N.lincomb(y, 2, Hydot, -2*store.f, ydot);
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

end
