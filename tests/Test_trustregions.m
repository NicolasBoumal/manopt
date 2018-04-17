function Test_trustregions

    
    % We attempt to compute an intrinsic mean of subspaces, that is, of
    % points on the Grassmann manifold. Let there be m subspaces of
    % dimension p embedded in R^n. We generate random data for our tests:
    % X is a random nxpxm matrix such that each slice of size nxp is an
    % orthonormal basis of a subspace.
    n = 50;
    p = 3;
    m = 100;
    Gr_multi = grassmannfactory(n, p, m);
    X = Gr_multi.rand();

    % Our search space is the Grassmann manifold: we want to locate one
    % point on the Grassmannian that "averages" the m given points.
    Gr = grassmannfactory(n, p);

    % The cost is the sum of squared Riemannian distances to the data
    % points.
    function f = cost(x)
        f = 0;
        for i = 1 : m
            xi = X(:, :, i);
            f = f + Gr.dist(x, xi)^2;
        end
        f = f/(2*m);
    end

    % The gradient is based on the Riemannian logarithmic map at the
    % current point, which gives tangent vectors at x pointing towards the
    % data points.
    function g = grad(x)
        g = Gr.zerovec(x);
        for i = 1 : m
            xi = X(:, :, i);
            g = g - Gr.log(x, xi);
        end
        g = g / m;
    end

    % Setup the problem structure, with the manifold M and the cost and its
    % gradient. Notice that we do not provide a Hessian, because it's
    % rather tricky to compute.
    problem.M = Gr;
    problem.cost = @cost;
    problem.grad = @grad;
%     problem.hess = @(x, xdot) Gr.zerovec(x) ;% * Gr.norm(x, xdot); 
    
    % For peace of mind, check that the gradient is correct.
    % checkgradient(problem);
    % pause;
    
    % As a simple minded initial guess, choose one of the data points.
    x0 = X(:, :, 1);
    
    % Setup some options for the trustregions algorihm
    options.tolgradnorm = 1e-16;
    options.maxtime = 30;
    options.maxiter = 200;
    options.verbosity = 2;
    options.debug = 0;
    options.rho_regularization = 1e3;
    
    % We did not specify a Hessian, but use trustregions anyway. Hence, the
    % Hessian will be approximated, and we should be warned about it. To
    % disable the warning, you may execute this command:
    warning('off', 'manopt:getHessian:approx');
    
    [x, cost_x, info] = trustregions(problem, x0, options); %#ok<ASGLU>
    
    xdata = [info.time];
    ydata = [info.gradnorm];
    semilogy(xdata, ydata, '.-');
    xlabel('Time [s]');
    ylabel('Gradient norm');
    hold on;
    radius_change = [0 sign(diff([info.Delta]))];
    text(xdata(radius_change > 0), ydata(radius_change > 0)/2, '+');
    text(xdata(radius_change < 0), ydata(radius_change < 0)/2, '-');
    text(xdata, ydata*2, num2str([info.numinner]'));
    hold off;
    
    % info
    % keyboard;

end