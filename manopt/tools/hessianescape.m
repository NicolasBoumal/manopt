function [y, lambda] = hessianescape(problem, x)

    % TODO : if really only want an escape direction, could use as a
    % stopping criterion that the cost is negative.

    % Define a manifold that is actually the unit sphere on the tangent
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
    % tangent vector to M at x.
    N = struct();
    % u, u1 and u2 will be tangent vectors to N at y. The tangent space to
    % N at y is a subspace of the tangent space to M at x, thus u, u1 and
    % u2 are also tangent vectors to M at x.
    
    N.dim = @() M.dim() - 1;
    N.inner = @(y, u1, u2) M.inner(x, u1, u2);
    N.norm  = @(y, u)      sqrt(N.inner(y, u, u));
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
    % Or ask for it in inputs?
    storedb = struct();
    if canGetGradient(problem)
        [unused1, unused2, storedb] = getCostGrad(problem, x, struct()); %#ok<ASGLU>
    end
    
    hessian = @(y) getHessian(problem, x, y, storedb);
    
    new_problem.M = N;
    new_problem.costgrad = @costgrad;
    function [val, grad] = costgrad(y)
        Hy = hessian(y);
        val = M.inner(x, y, Hy);
        grad = M.lincomb(y, 2, Hy, -2*val, y);
    end
    
    % checkgradient(new_problem);
    
    % TODO: have default options and accept options as input to merge with
    % the defaults.
    options.tolcost = -eps;
    options.tolgradnorm = 1e-6;
    
    [y, lambda] = conjugategradient(new_problem, [], options);

end
