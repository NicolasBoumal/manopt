function Test_nonlinear_eigenvalue_problem()
    
    % Problem instance.
    n = 1000;
    L = gallery('tridiag', n, -1, 2, -1);
    k = 10;
    alpha = 1;
    
    % Grassmann manifold description.
    Gr = grassmannfactory(n, k);
    problem.M = Gr;
    
    % Cost function evaluation
    problem.cost =  @cost;
    function val = cost(X)
        rhoX = sum(X.^2, 2); %diag(X*X'); 
        val = 0.5*trace(X'*(L*X)) + (alpha/4)*(rhoX'*(L\rhoX));
    end
    
    % Euclidean gradient evaluation.
    % Note: Manopt automatically converts it to the Riemannian counterpart.
    problem.egrad = @egrad;
    function g = egrad(X)
        rhoX = sum(X.^2, 2); %diag(X*X');
        g = L*X + alpha*diag(L\rhoX)*X;
        
    end
    
    % Euclidean Hessian evaluation.
    % Note: Manopt automatically converts it to the Riemannian counterpart.
    problem.ehess = @ehess;
    function h = ehess(X, U)
        rhoX = sum(X.^2, 2); %diag(X*X');
        rhoXdot = 2*sum(X.*U, 2); 
        h = L*U + alpha*diag(L\rhoXdot)*X + alpha*diag(L\rhoX)*U;
    end
    
    
    % Check whether gradient and Hessian computations are correct
    checkgradient(problem);
    pause;
    checkhessian(problem);
    pause;
    
    
    % Call the trust-region algorithm.
    Xsol = trustregions(problem);
    
end
