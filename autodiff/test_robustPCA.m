function  test_gradient_robustpca()
% Main author: Nicolas Boumal, July 5, 2013

    import casadi.*
    X = rand(2, 1)*(1:30) + .05*randn(2, 30).*[(1:30);(1:30)];
    P = randperm(size(X, 2));
    outliers = 10;  
    X(:, P(1:outliers)) = 30*randn(2, outliers);
    x = randn(2,1);
    xdot = randn(2,1);
    epsilon = 1e-5;
    
    mycostfunction = @robustpca_cost;
    g_auto = auto_grad(x,mycostfunction);
    h_auto = auto_hess(x,xdot,mycostfunction);
    
    manifold = grassmannfactory(2, 1);
    problem.M = manifold;
    problem.cost = @robustpca_cost;
    problem.egrad = @(x) auto_grad_call(x,g_auto);
    problem.ehess = @(x,xdot) auto_hess_call(x,xdot,h_auto);
    
    figure;
    checkgradient(problem);
    figure;
    checkhessian(problem);
    
    function value = robustpca_cost(U)

        vecs = U*(U'*X) - X;
        sqnrms = sum(vecs.^2, 1);
        vals = sqrt(sqnrms + epsilon^2) - epsilon;
        value = mean(vals);
        
    end

    [U, ~, ~] = svds(X, 1);
    [U, Ucost, info] = trustregions(problem,U);
    figure;
    semilogy([info.iter], [info.gradnorm], '.-');
    xlabel('Iteration #');
    ylabel('Gradient norm');
    title('Convergence of the trust-regions algorithm');

end
