function test19()
% Test for complexcircle geometry (Phaseless signal reconstruction)
% Based on formulation by Waldspurger, d'Aspremont and Mallat, 2012, eq (4)
%

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 


    reset(RandStream.getDefaultStream);
    randnfoo = randn(12345, 1); %#ok<NASGU>

    % signal length
    n = 128;
    % number of measurements
    m = 4*n;
    
    % measurement matrix (each row has unit-norm)
    A = randn(m, n) + 1i*randn(m, n);
    A = A ./ repmat(sqrt(sum(A.*conj(A), 2)), 1, n);
    
    % true signal
    x = randn(n, 1) + 1i*randn(n, 1);
    
    % measurement vector
    b_true = abs(A*x);
%     hist(b_true);
%     pause;
%     b = exprnd(b_true);
    b = max(0.01, b_true + .2*randn(m, 1));
%     b = b_true;
%     b = b_true .* exprnd(1, m, 1);
%     plot(sort(b));
%     pause;
    
    % quadratic form for phaseless reconstruction
    % (this is a bad way of representing this operator of course)
%     M = diag(b)*(eye(m) - A*pinv(A))*diag(b);
%     Mop = @(x) M*x;
    
    % Alternate definition as operator. This is equivalent to the former
    % definition but we just need to precompute a basis for the range of A.
    % Applying A*pinv(A) to a vector then comes down to projection of that
    % vector on the range space of A, which is quick to do. I-A*pinv(A) is
    % the projection on the horizontal space to the range.
    Mop = @Mfun;
    % [U, ~, ~] = svd(A, 'econ');
    [U, ~] = qr(A, 0);
    function y = Mfun(y)
        y = b.*y;
        y = y - U*(U'*y);
        y = b.*y;
    end
    
    % Pick a manifold
    problem.M = complexcirclefactory(m);
    
    % Define cost and gradient
    problem.cost = @cost;
    function [f, store] = cost(u, store)
        if ~isfield(store, 'Mu')
            store.Mu = Mop(u);
        end
        Mu = store.Mu;
        f = .5*real(u'*Mu);
    end

    problem.grad = @grad;
    function [g, store] = grad(u, store)
        if ~isfield(store, 'Mu')
            store.Mu = Mop(u);
        end
        Mu = store.Mu;
        g = problem.M.proj(u, Mu);
    end

    problem.hess = @hess;
    function [h, store] = hess(u, xi, store)
        if ~isfield(store, 'Mu')
            store.Mu = Mop(u);
        end
        Mu = store.Mu;
        Mxi = Mop(xi);
        h = problem.M.proj(u, Mxi - real(conj(Mu).*u).*xi ...
                                  - real(conj(Mxi).*u + conj(Mu).*xi).*u);
    end

    problem.precon = @precon;
    function [h, store] = precon(u, xi, store) %#ok<INUSL>
        h = xi./(b.^2);
    end
    
%     checkgradient(problem);
%     pause;
%     checkhessian(problem);
%     pause;
    
%     z0 = (A*x)./abs(A*x);

    z0cost = inf;
    z0 = [];
    for i = 1 : 100
        z0trial = problem.M.rand();
        cst = cost(z0trial, struct());
        if cst < z0cost
            z0cost = cst;
            z0 = z0trial;
        end
    end

    options.Delta_bar = 10*pi*n;
    options.maxiter = inf;
    options.maxinner = 1000;
    options.maxtime = 300;
    options.storedepth = 10;
    profile clear;
    profile on;
    [zopt, costopt, info] = trustregions(problem, z0, options); %#ok<ASGLU>
%     [zopt costopt info] = conjugategradient(problem, z0, options);
    profile off;
    profile report;
    
    % Checkout the Hessian spectrum with and without preconditioner.
    % First with the preconditioner as is in the problem structure
    tic;
    eig_precon = hessianspectrum(problem, zopt);
    toc
    % Then, without the preconditioner: we remove it from the structure.
    problem = rmfield(problem, 'precon');
    tic;
    eig_noprecon = hessianspectrum(problem, zopt);
    toc
    % Finally, with the square root preconditioner
    % (as an operator): providing it explicitly can sometimes speed up the
    % computation because it allows to compute the spectrum via a symmetric
    % operator.
    problem.sqrtprecon = @sqrtprecon;
    function [h, store] = sqrtprecon(u, xi, store) %#ok<INUSL>
        h = xi./b;
    end
    tic;
    eig_precon_sqrt = hessianspectrum(problem, zopt);
    toc
    disp([eig_precon eig_precon_sqrt eig_noprecon]);
    
    
    xopt = A\(zopt.*b);
    
    q = sqrt(abs(xopt'*x))/norm(x);
    fprintf('Recovery quality: %g\n1-recovery quality: %g\n', q, 1-q);

%     keyboard;

end
