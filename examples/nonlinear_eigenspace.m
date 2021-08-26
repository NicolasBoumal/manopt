function Xsol = nonlinear_eigenspace(L, k, alpha)
% Example of nonlinear eigenvalue problem: total energy minimization.
%
% function Xsol = nonlinear_eigenspace(L, k, alpha)
%
% L is a discrete Laplacian operator,
% alpha is a given constant, and
% k corresponds to the dimension of the least eigenspace sought. 
%
% This example demonstrates how to use the Grassmann geometry factory 
% to solve the nonlinear eigenvalue problem as the optimization problem:
%
% minimize 0.5*trace(X'*L*X) + (alpha/4)*(rho(X)*L\(rho(X))) 
% over X such that X'*X = Identity,
%
% where L is of size n-by-n,
% X is an n-by-k matrix, and
% rho(X) is the diagonal part of X*X'.
%
% This example is motivated in the paper
% "A Riemannian Newton Algorithm for Nonlinear Eigenvalue Problems",
% Zhi Zhao, Zheng-Jian Bai, and Xiao-Qing Jin,
% SIAM Journal on Matrix Analysis and Applications, 36(2), 752-774, 2015.
%


% This file is part of Manopt and is copyrighted. See the license file.
%
% Main author: Bamdev Mishra, June 19, 2015.
% Contributors:
%
% Change log:
%
%    Aug. 20, 2021(XJ)
%       Added AD to compute the egrad and the ehess 

    % If no inputs are provided, generate a  discrete Laplacian operator.
    % This is for illustration purposes only.
    % The default example corresponds to Case (c) of Example 6.2 of the
    % above referenced paper.
    
    if ~exist('L', 'var') || isempty(L)
        n = 100;
        L = gallery('tridiag', n, -1, 2, -1);
    end
    
    n = size(L, 1);
    assert(size(L, 2) == n, 'L must be square.');
    
    if ~exist('k', 'var') || isempty(k) || k > n
        k = 10;
    end
    
    if ~exist('alpha', 'var') || isempty(alpha)
        alpha = 1;
    end
    
    
    % Grassmann manifold description
    Gr = grassmannfactory(n, k);
    problem.M = Gr;
    
    % Cost function evaluation
    problem.cost =  @cost;
    function val = cost(X)
        rhoX = sum(X.^2, 2); % diag(X*X'); 
        val = 0.5*trace(X'*(L*X)) + (alpha/4)*(rhoX'*(L\rhoX));
    end
    
    % Euclidean gradient evaluation
    % Note: Manopt automatically converts it to the Riemannian counterpart.
    problem.egrad = @egrad;
    function g = egrad(X)
        rhoX = sum(X.^2, 2); % diag(X*X');
        g = L*X + alpha*diag(L\rhoX)*X;
    end
    
    % Euclidean Hessian evaluation
    % Note: Manopt automatically converts it to the Riemannian counterpart.
    problem.ehess = @ehess;
    function h = ehess(X, U)
        rhoX = sum(X.^2, 2); %diag(X*X');
        rhoXdot = 2*sum(X.*U, 2); 
        h = L*U + alpha*diag(L\rhoXdot)*X + alpha*diag(L\rhoX)*U;
    end
    
    % An alternatie way to compute the egrad and the ehess is to use 
    % automatic differentiation provided in the deep learning tool box(slower)
    % Notice that the function trace is not supported for AD so far.
    % Replace it with ctrace described in the file functions_AD.m
    % Also, operations between sparse matices and dlarrys are not
    % supported. Convert L into a full matrix for the use of AD.
    % The operation \ is not supported for AD. Convert it to inv()*
    % L_full = full(L);
    % problem.cost = @cost_AD;
    %    function val = cost_AD(X)
    %        rhoX = sum(X.^2, 2); % diag(X*X'); 
    %        val = 0.5*ctrace(X'*(L_full*X)) + (alpha/4)*(rhoX'*(inv(L_full)*rhoX));
    %    end
    % call preprocessAD to automatically obtain the egrad and the ehess
    % problem = preprocessAD(problem);

    % Check whether gradient and Hessian computations are correct.
    % checkgradient(problem);
    % pause;
    % checkhessian(problem);
    % pause;
    
    
    % Initialization as suggested in above referenced paper.
    X = randn(n, k);
    [U, S, V] = svd(X, 0); %#ok<ASGLU>
    X = U*V';
    [U0, S0, V0] = eigs(L + alpha*diag(L\(sum(X.^2, 2))), k,'sm'); %#ok<NASGU,ASGLU>
    X0 = U0;
  
    % Call manoptsolve to automatically call an appropriate solver.
    % Note: it calls the trust regions solver as we have all the required
    % ingredients, namely, gradient and Hessian, information.
    Xsol = manoptsolve(problem, X0);
    
end
