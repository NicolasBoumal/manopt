function test_simple()
% Main author: Nicolas Boumal, July 5, 2013
    n=10000;
    A = randn(n);
    A = .5*(A+A');
    x = randn(n,1);
    xdot = randn(n,1);
    manifold = spherefactory(n);
    
    mycostfunction = @(x) -x'*(A*x);
    g_auto = auto_grad(x,mycostfunction);
    h_auto = auto_hess(x,xdot,mycostfunction);
    
    problem.cost = @(x) mycostfunction(x); 
    problem.M = manifold;
    problem.egrad = @(x) auto_grad_call(x,g_auto);
    problem.ehess = @(x,xdot) auto_hess_call(x,xdot,h_auto);

    figure;
    checkgradient(problem);
    figure;
    checkhessian(problem);

    % Solve.
    [x, xcost, info] = steepestdescent(problem);         
    
    % Display some statistics.
    figure;
    semilogy([info.iter], [info.gradnorm], '.-');
    xlabel('Iteration #');
    ylabel('Gradient norm');
    title('Convergence of the trust-regions algorithm on the sphere');
    
    g_auto
end


