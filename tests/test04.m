function [Xlsq, Xhub, D] = test04(n, m)
% function [Xlsq Xhub D] = test04(n, m)
% All intputs are optional.
%
% Crude formulation of the photometric stereo problem to experiment with
% product manifolds: we are looking for two sets of normal vectors in R^3,
% more precisely, we work on the product OB(3, m) x OB(3, n).
% In practice, we pick two such fields at random and form their m-by-n
% product, then attempt to factor the product back into the original two
% fields. Of course, this is up to a global rotation of both fields.
%
% We try two approaches:
%  1) minimizing a least-squares criterion
%  2) minimizing a Huber loss (lsq is a particular limit case of it)
%

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 

    warning('off', 'manopt:getHessian:approx');
    
    if ~exist('n', 'var') || isempty(n)
        n = 6;
    end
    if ~exist('m', 'var') || isempty(m)
        m = 100;
    end

    % Create the problem structure
    Nspace = obliquefactory(3, m);
    Lspace = obliquefactory(3, n);
    NLspace = productmanifold(struct('N', Nspace, 'L', Lspace));
    problem.M = NLspace;

    % Define the problem data
    sol = NLspace.rand();
    D = sol.N' * sol.L;
%     D = D + .01*randn(size(D));
    sub = rand(size(D)) < 0.10;
    D(sub) = rand(nnz(sub), 1)*2-1;
    
    % Everybody starts from the same point.
    X0 = NLspace.rand();

    %% Least-squares formulation
    
    problem.cost = @(X) .5*norm(X.N'*X.L-D, 'fro')^2;
    problem.grad = @(X) NLspace.proj(X, struct('N', X.L*(X.N'*X.L-D)', 'L', X.N*(X.N'*X.L-D)));
    
    % Check gradient consistency.
    checkgradient(problem);

    % Solve
    Xlsq = trustregions(problem, sol);
    
    
    %% Huber loss formulation
    a = .01;
    
    problem.cost = @hubercost;
    function val = hubercost(X)
        residue = X.N'*X.L - D;
        mask = abs(residue) < a;
        val = .5*sum(residue(mask(:)).^2) + a*sum(abs(residue(~mask(:))) - a/2);
    end
    
    problem.grad = @(X) NLspace.proj(X, hubergrad(X));
    function g = hubergrad(X)
        residue = X.N'*X.L - D;
        clipped = min(max(residue, -a), a);
        g.N = X.L*clipped';
        g.L = X.N*clipped;
    end
    
    % Check gradient consistency.
    checkgradient(problem);

    % Solve
    Xhub = trustregions(problem, sol);
    
    
    fprintf('Least-squares dist to solution: %g.\n', norm(sol.N'*sol.L - Xlsq.N'*Xlsq.L, 'fro')/sqrt(m*n));
    fprintf('Huber loss    dist to solution: %g.\n', norm(sol.N'*sol.L - Xhub.N'*Xhub.L, 'fro')/sqrt(m*n));
    
    % Add a little Nelder-Mead testing
    optionsnm.maxcostevals = problem.M.dim() * 10;
    optionsnm.maxiter = 2000;
    simplex = cell(problem.M.dim()+1, 1);
    simplex{1} = problem.M.rand();
    for i = 2 : numel(simplex)
        simplex{i} = problem.M.retr(simplex{1}, problem.M.randvec(simplex{1}), .1);
    end
    neldermead(problem, simplex, optionsnm);
    
end
