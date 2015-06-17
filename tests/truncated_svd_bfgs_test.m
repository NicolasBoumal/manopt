function truncated_svd_bfgs_test(A, p)
    
    % Generate some random data to test the function if none is given.
    if ~exist('A', 'var') || isempty(A)
        A = randn(402, 500);
    end
    if ~exist('p', 'var') || isempty(p)
        p = 5;
    end
    
    % Retrieve the size of the problem and make sure the requested
    % approximation rank is at most the maximum possible rank.
    [m, n] = size(A);
    assert(p <= min(m, n), 'p must be smaller than the smallest dimension of A.');
    
    % Define the cost and its derivatives on the Grassmann manifold

    tuple.U = grassmannfactory(m, p);
    tuple.V = grassmannfactory(n, p);
    
    M = productmanifold(tuple);
    
    problem.M = M;
    problem.cost  = @cost;
    problem.egrad = @egrad;
    problem.ehess = @ehess;

    % Cost function
    function f = cost(X)
        U = X.U;
        V = X.V;
        f = -.5*norm(U'*A*V, 'fro')^2;
    end
    
    % Euclidean gradient of the cost function
    function g = egrad(X)
        U = X.U;
        V = X.V;
        AV = A*V;
        AtU = A'*U;
        g.U = -AV*(AV'*U);
        g.V = -AtU*(AtU'*V);
    end
    
    % Euclidean Hessian of the cost function
    function h = ehess(X, H)
        U = X.U;
        V = X.V;
        Udot = H.U;
        Vdot = H.V;
        AV = A*V;
        AtU = A'*U;
        AVdot = A*Vdot;
        AtUdot = A'*Udot;
        h.U = -(AVdot*AV'*U + AV*AVdot'*U + AV*AV'*Udot);
        h.V = -(AtUdot*AtU'*V + AtU*AtUdot'*V + AtU*AtU'*Vdot);
    end
    
    conjugategradient(problem);
    %     trustregions(problem);
    
    problem.precon = preconBFGS(problem);
    problem.linesearch = @(x, xdot, storedb, key) 1;
    options.beta_type ='steep';
    conjugategradient(problem, [],options);
end
