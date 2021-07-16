function truncated_SVD()

% Main author: Nicolas Boumal, July 5, 2013

    A = randn(42, 60);
    p = 5;
    % Test
    X.U = randn(42,5);
    X.V = randn(60,5);
    mycostfunction = @cost;
    g_auto = auto_grad(X,mycostfunction);
    
    % Define the cost and its derivatives on the Grassmann manifold
    tuple.U = grassmannfactory(42, 5);
    tuple.V = grassmannfactory(60, 5);
    M = productmanifold(tuple);
    
    problem.M = M;
    problem.cost  = @cost;
    problem.egrad = @(X) auto_grad_call(X,g_auto);  
    
    checkgradient(problem);
    
    function f = cost(X)
        U = X.U;
        V = X.V;
        f = -.5*norm(U'*A*V, 'fro')^2;
    end
    
end
