function [] = test10()
% function test10()
%
% Test the fixed rank geometry with a quick and dirty low rank matrix
% completion problem.
%

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 


%%%% You need to change the fixedrankembeddedfactory:
%     function X = random()
%         X.U = stiefelm.rand();
%         X.V = stiefeln.rand();
%         X.S = diag(randn(k,1));   % <----- Here :-)
%     end
    
    % Generate the data
    m = 500;
    n = 400;
    k = 10;
    L = randn(m, k);          % generate a random mxn matrix of rank k
    R = randn(n, k);
    A = L*R';
    % generate a random mask for observed entries
    P = sparse(round(.75*rand(m, n)));
    
    % Pick the manifold
    problem.M = fixedrankembeddedfactory(m, n, k);

    % Define the problem cost function
    problem.cost = @(X) .5*norm(P.*(X.U*X.S*X.V'-A), 'fro')^2;

    % And its gradient
    problem.egrad = @(X) P.*(X.U*X.S*X.V'-A);

    % And its Hessian
    problem.ehess = @(X, H) P.*problem.M.tangent2ambient(X, H);
    

    % Check differentials consistency.
    warning('off', 'manopt:fixedrank:exp');
    checkgradient(problem); pause;
    checkhessian(problem); pause;
    
    
    problem.hess = @Riemannian_Hessian;
    function Z = Riemannian_Hessian(X, D)
        G = P.*(X.U*X.S*X.V'-A);
        H1_D = P.*problem.M.tangent2ambient(X, D);
        Z = problem.M.proj(X, H1_D);
        
        T = (G*D.Vp)/X.S;
        Z.Up = Z.Up + (T - X.U*(X.U'*T));
        
        T = (G'*D.Up)/X.S;
        Z.Vp = Z.Vp + (T - X.V*(X.V'*T));
    end

    % Check differentials consistency for supplied Hessian.
    checkhessian(problem); pause;
    
    

    % Solve with trust-regions and FD approximation of the Hessian
%     warning('off', 'manopt:getHessian:approx');
    

    %%% X.S should be diagonal!!!!!   % < -----
    %[U, S, V] = svds(P.*A, k);
    [UU,SS,VV] = svd(full(P.*A));
    X.U = UU(:,1:k); X.S = SS(1:k,1:k); X.V = VV(:,1:k);
    options.maxiter = 500;
%     options.Delta_bar = 2000;
%     options.Delta0 = options.Delta_bar / 8;
    X = trustregions(problem, X, options);
    
    fprintf('||X-A||_F = %g\n', norm(X.U*X.S*X.V' - A, 'fro'));
    
%     conjugategradient(problem);
%     steepestdescent(problem);
    
%     keyboard;

%     hist(hessianspectrum(problem, X));
    
end
