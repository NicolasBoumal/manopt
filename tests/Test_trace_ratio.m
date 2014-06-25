function Test_trace_ratio()
    %
    % Maximize trace(X'*A*X)/trace(X'*B*X) with B being symmetric and positive
    % semidefinite and A is only symmetric.
    %
    
    % This file is part of Manopt and is copyrighted. See the license file.
    %
    % Main author: Bamdev Mishra
    % Contributors:
    %
    % Change log:
    %
 
    n = 200;
    p = 5;
    A = randn(n, n);
    A = A + A'; % Symmetric matrix
    B = randn(n, n - p + 1);
    B = B*B'; % Symmetric positive semidefinite
    
    
    % Define the cost and its derivatives on the Grassmann manifold
    Gr = grassmannfactory(n, p);
    problem.M = Gr;
    
    problem.cost = @cost;
    function[val, store] = cost(X, store)
        if ~all(isfield(store,{'AX','BX','traceXAX','traceXBX'}))
            store.AX = A*X;
            store.BX = B*X;
            store.traceXAX = trace(X'*store.AX);
            store.traceXBX = trace(X'*store.BX);
        end
        
        traceXAX = store.traceXAX;
        traceXBX = store.traceXBX;
        
        val = -0.5*traceXAX/traceXBX;
    end
    
    problem.grad = @grad;
    function [Rgrad store] = grad(X, store)
        if ~all(isfield(store,{'AX','BX','traceXAX','traceXBX'}))
            store.AX = A*X;
            store.BX = B*X;
            store.traceXAX = trace(X'*store.AX);
            store.traceXBX = trace(X'*store.BX);
        end
        
        AX = store.AX;
        BX = store.BX;
        traceXAX = store.traceXAX;
        traceXBX = store.traceXBX;
        
        % Euclidean gradient
        egrad = - (traceXBX*AX - traceXAX*BX)/(traceXBX ^2) ;
        store.egrad = egrad; % store Euclidean gradient
        
        % Euclidean gradient to Riemannian gradient
        Rgrad = Gr.egrad2rgrad(X, egrad);
    end
    
    
    problem.hess = @hess;
    function [RHess store] = hess(X, eta, store)
        
        if ~all(isfield(store,{'AX','BX','traceXAX','traceXBX'}))
            store.AX = A*X;
            store.BX = B*X;
            store.traceXAX = trace(X'*store.AX);
            store.traceXBX = trace(X'*store.BX);
        end
        
        AX = store.AX;
        BX = store.BX;
        traceXAX = store.traceXAX;
        traceXBX = store.traceXBX;
        Aeta = A*eta;
        Beta = B*eta;
        
        % Euclidean gradient
        if ~ isfield(store, 'egrad');
            store.egrad = - (traceXBX*AX - traceXAX*BX)/(traceXBX ^2);
        end
        egrad = store.egrad;
        
        
        % Euclidean Hessian
        eHess = -Aeta/traceXBX + 2*trace(eta'*BX)*AX /(traceXBX ^2);
        eHess = eHess + (traceXAX*Beta + 2*trace(eta'*AX)*BX )/(traceXBX ^2)...
            - 4*traceXAX*trace(eta'*BX)*BX/(traceXBX ^3);
        
        % Euclidean Hessian to Riemannian Hessian
        RHess = Gr.ehess2rhess(X, egrad, eHess, eta);
    end
    
    % Execute some checks on the derivatives for early debugging.
    % These things can be commented out of course.
    checkgradient(problem);
    pause;
    checkhessian(problem);
    pause;
    
    % Issue a call to a solver. A random initial guess will be chosen and
    % default options are selected.
    options.maxinner = 500;
    Xsol = trustregions(problem,[], options);
    
    rho = trace(Xsol'*(A*Xsol))/trace(Xsol'*(B*Xsol));
    trace(Xsol'*((A - rho*B)*Xsol)) % Global optimality certificate
end
