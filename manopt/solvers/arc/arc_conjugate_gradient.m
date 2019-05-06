function [eta, Heta, hesscalls, stop_str, stats] = arc_conjugate_gradient(problem, x, grad, gradnorm, sigma, options, storedb, key)
% Subproblem solver for ARC based on a nonlinear conjugate gradient method.
%
% [eta, Heta, hesscalls, stop_str, stats] = 
%     arc_conjugate_gradient(problem, x, grad, gradnorm, sigma, options, storedb, key)
%
% This routine approximately solves the following problem:
%
%   min_{eta in T_x M}  m(eta),  where
%
%       m(eta) = <eta, g> + .5 <eta, H[eta]> + (sigma/3) ||eta||^3
%
% where eta is a tangent vector at x on the manifold given by problem.M,
% g = grad is a tangent vector at x, H[eta] is the result of applying the
% Hessian of the problem at x along eta, and the inner product and norm
% are those from the Riemannian structure on the tangent space T_x M.
%
% The solve is approximate in the sense that the returned eta only ought
% to satisfy the following conditions:
%
%   ||gradient of m at eta|| <= theta*||eta||^2   and   m(eta) <= m(0),
%
% where theta is specified in options.theta (see below for default value.)
% Since the gradient of the model at 0 is g, if it is zero, then eta = 0
% is returned. This is the only scenario where eta = 0 is returned.
%
% Numerical errors can perturb the described expected behavior.
%
% Inputs:
%     problem: Manopt optimization problem structure
%     x: point on the manifold problem.M
%     grad: gradient of the cost function of the problem at x
%     gradnorm: norm of the gradient, often available to the caller
%     sigma: cubic regularization parameter (positive scalar)
%     options: structure containing options for the subproblem solver
%     storedb, key: caching data for problem at x
%
% Options specific to this subproblem solver:
%   theta (0.25)
%     Stopping criterion parameter for subproblem solver: the gradient of
%     the model at the returned step should have norm no more than theta
%     times the squared norm of the step.
%   maxiter_cg (500)
%     Maximum number of iterations of the conjugate gradient algorithm.
%   beta_type ('P-R')
%     The update rule for calculating beta:
%     'F-R' for Fletcher-Reeves, 'P-R' for Polak-Ribiere, and 'H-S' for
%     Hestenes-Stiefel.
%
% Outputs:
%     eta: approximate solution to the cubic regularized subproblem at x
%     Heta: Hess f(x)[eta] -- this is necessary in the outer loop, and it
%           is often naturally available to the subproblem solver at the
%           end of execution, so that it may be cheaper to return it here.
%     hesscalls: number of Hessian calls during execution
%     stop_str: string describing why the subsolver stopped
%     stats: a structure specifying some statistics about inner work
%            (currently unused)

% This file is part of Manopt: www.manopt.org.
% Original authors: May 2, 2019,
%    Bryan Zhu, Nicolas Boumal.
% Contributors:
% Change log: 

    % Some shortcuts
    M = problem.M;
    n = M.dim();
    inner = @(u, v) M.inner(x, u, v);
    rnorm = @(u) M.norm(x, u);
    tangent = @(u) problem.M.tangent(x, u);
    Hess = @(u) getHessian(problem, x, u, storedb, key);
    
    % Counter for Hessian calls issued
    hesscalls = 0;
    
    % If the gradient has norm zero, return a zero step
    if gradnorm == 0
        eta = M.zerovec(x);
        Heta = eta;
        stop_str = 'Cost gradient is zero';
        stats = struct('gradnorms', 0, 'func_values', 0);
        return;
    end
    
    % Set local defaults here
    localdefaults.theta = 0.25;
    localdefaults.maxiter_cg = 500;
    localdefaults.beta_type = 'P-R';
    
    % Merge local defaults with user options, if any
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    % Calculate the Cauchy point as our initial step
    hess_grad = Hess(grad);
    hesscalls = hesscalls + 1;
    temp = inner(grad, hess_grad) / (2 * sigma * gradnorm * gradnorm);
    R_c = -temp + sqrt(temp * temp + gradnorm / sigma);
    eta = M.lincomb(x, -R_c / gradnorm, grad);
    Heta = M.lincomb(x, -R_c / gradnorm, hess_grad);
    
    % Initialize variables needed for calculation of conjugate direction
    prev_grad = M.lincomb(x, -1, grad);
    prev_conj = prev_grad;
    Hp_conj = M.lincomb(x, -1, hess_grad);
    
    % Main conjugate gradients iteration
    maxiter = min(options.maxiter_cg, n);
    gradnorms = zeros(maxiter, 1);
    func_values = zeros(maxiter, 1);
    gradnorm_reached = false;
    j = 1;
    while j < maxiter
        % Calculate the gradient
        eta_norm = rnorm(eta);
        new_grad = M.lincomb(x, 1, Heta, 1, grad);
        new_grad = M.lincomb(x, -1, new_grad, -sigma * eta_norm, eta);
        new_grad = tangent(new_grad);
        gradnorms(j) = rnorm(new_grad);
        func_values(j) = inner(grad, eta) + 0.5 * inner(eta, Heta) + (sigma/3) * eta_norm^3;
        
        if options.verbosity >= 4
            fprintf('\nModel grad norm: %.16e, Iterate norm: %.16e', rnorm(new_grad), eta_norm);
        end

        % Check termination condition
        if rnorm(new_grad) <= options.theta * eta_norm^2
            stop_str = 'Model grad norm condition satisfied';
            gradnorm_reached = true;
            break;
        end
        
        % Calculate the conjugate direction using the selected beta rule
        delta = M.lincomb(x, 1, new_grad, -1, prev_grad);
        switch upper(options.beta_type)
            case 'F-R'
                beta = inner(new_grad, new_grad) / inner(prev_grad, prev_grad);
            case 'P-R'
                beta = max(0, inner(new_grad, delta) / inner(prev_grad, prev_grad));
            case 'H-S'
                beta = max(0, -inner(new_grad, delta) / inner(prev_conj, delta));
            otherwise
                error('Unknown options.beta_type. Should be F-R, P-R, or H-S.');
        end
        new_conj = M.lincomb(x, 1, new_grad, beta, prev_conj);
        Hn_grad = Hess(new_grad);
        hesscalls = hesscalls + 1;
        Hn_conj = M.lincomb(x, 1, Hn_grad, beta, Hp_conj);
        
        % Find the optimal step in the conjugate direction
        alpha = solve_along_line(M, x, eta, new_conj, grad, Hn_conj, sigma);
        if alpha == 0
            stop_str = 'Reached optimum within search direction';
            gradnorm_reached = true;
            break;
        end
        eta = M.lincomb(x, 1, eta, alpha, new_conj);
        Heta = M.lincomb(x, 1, Heta, alpha, Hn_conj);
        prev_grad = new_grad;
        prev_conj = new_conj;
        Hp_conj = Hn_conj;
        j = j + 1;
    end
    
    % Check why we stopped iterating
    if ~gradnorm_reached
        stop_str = sprintf(['Reached max number of conjugate gradient iterations ' ...
               '(options.maxiter_cg = %d)'], options.maxiter_cg);
        j = j - 1;
    end
    
    % Return the point we ended on
    eta = tangent(eta);
    stats = struct('gradnorms', gradnorms(1:j), 'func_values', func_values(1:j));
    if options.verbosity >= 4
        fprintf('\n');
    end
        
end
