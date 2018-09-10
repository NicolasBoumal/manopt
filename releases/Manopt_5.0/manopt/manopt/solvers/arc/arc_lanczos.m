function [eta, Heta, hesscalls, stop_str, stats] = arc_lanczos(problem, x, grad, gradnorm, sigma, options, storedb, key)
% Subproblem solver for ARC based on a Lanczos process.
%
% [eta, Heta, hesscalls, stop_str, stats] = 
%     arc_lanczos(problem, x, grad, gradnorm, sigma, options, storedb, key)
%
% This routine approximately solves the following problem:
%
%   min_{eta in T_x M}  m(eta),  where
%
%       m(eta) = <eta, g> + .5 <eta, H[eta]> + (sigma/3) ||eta||^3
%
% where eta is a tangent vector at x on the manifold given by problem.M,
% g = grad is a tangent vector at x, H[eta] is the result of applying the
% Hessian of the problem at x along eta and the inner product and norm
% are those from the Riemannian structure on the tangent space T_x M.
%
% The solve is approximate in the sense that the returned s only ought to
% satisfy the following conditions:
%
%     ||gradient of m at s|| <= theta*||s||^2   and   m(s) <= m(0),
%
% where theta is specified in options.theta (see below for default value.)
% Since the gradient of the model at 0 is g, if it is zero, then s = 0 is
% returned. This is the only scenario where s = 0 is returned.
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
%   theta (50)
%     Stopping criterion parameter for subproblem solver: the gradient of
%     the model at the returned step should have norm no more than theta
%     times the squared norm of the step.
%   maxiter_lanczos (M.dim())
%     Maximum number of iterations of the Lanczos process, which is nearly
%     the same as the maximum number of calls to the Hessian.
%   maxiter_newton (100)
%     Maximum number of iterations of the Newton root finder to solve each
%     tridiagonal cubic problem.
%   tol_newton (1e-16)
%     Tolerance for the Newton root finder.
%
% Outputs:
%     eta: approximate solution to the cubic regularized subproblem at x
%     Heta: Hess f(x)[eta] -- this is necessary in the outer loop, and it
%           is often naturally available to the subproblem solver at the
%           end of execution, so that it may be cheaper to return it here.
%     hesscalls: number of Hessian calls during execution
%     stop_str: string describing why the subsolver stopped
%     stats: a structure specifying some statistics about inner work
%
% See also: arc minimize_cubic_newton

% This file is part of Manopt: www.manopt.org.
% Original authors: May 1, 2018,
%    Naman Agarwal, Brian Bullins, Nicolas Boumal and Coralia Cartis.
% Contributors: 
% Change log: 

% TODO: think whether we can save the Lanczos basis in the storedb at the
% given key in case we get a rejection, and simply "start where we left
% off" with the updated sigma.

% TODO: Lanczos is notoriously numerically unstable, with loss of
% orthogonality being a main hurdle. Look into the literature (Paige,
% Parlett), for possible numerical fixes.

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
        stats = struct('newton_iterations', 0);
        return;
    end
    
    % Set local defaults here
    localdefaults.theta = 50;
    localdefaults.maxiter_lanczos = n;
    % The following are here for the Newton solver called below
    localdefaults.maxiter_newton = 100;
    localdefaults.tol_newton = 1e-16;
    
    % Merge local defaults with user options, if any
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    % Vector where we keep track of the Newton root finder's work
    newton_iterations = zeros(n, 1);
    
    % Lanczos iteratively produces an orthonormal basis of tangent vectors
    % which tridiagonalize the Hessian. The corresponding tridiagonal
    % matrix is preallocated here as a sparse matrix.
    T = spdiags(ones(n, 3), -1:1, n, n);
    
    % The orthonormal basis (n tangent vectors at x) is stored in this cell
    Q = cell(n, 1);
    
    % Initialize Lanczos along the gradient direction (it is nonzero)
    q = M.lincomb(x, 1/gradnorm, grad);
    Q{1} = q;
    Hq = Hess(q);
    hesscalls = hesscalls + 1;
    alpha = inner(Hq, q);
    T(1, 1) = alpha;
    Hq_perp = M.lincomb(x, 1, Hq, -alpha, q);
    
    % Minimizing the cubic restricted to the one-dimensional space spanned
    % by Q{1} is easy: it amounts to minimizing a univariate cubic. Indeed,
    % with eta = y*q where y is a scalar, we minimize (since g = ||g||q):
    %  h(y) = <y*q, g> + .5 <y*q, H[y*q]> + (sigma/3) ||y*q||^3
    %       = ||g||*y + .5*alpha*y^2 + (sigma/3) |y|^3.
    % The sign of y affects only the linear term, hence it is clear we need
    % to pick y nonpositive. In that case, h becomes a cubic polynomial:
    %  h(y) = ||g||*y + .5*alpha*y^2 - (sigma/3) y^3
    % The derivative is a quadratic polynomial:
    %  h'(y) = ||g|| + alpha*y - sigma*y^2.
    % Since ||g|| and sigma are positive, h' has two real roots, one
    % posivite and one negative (strictly). The negative root is the only
    % root of interest. It necessarily identifies a minimizer since
    % h(0) = 0, h(-inf) = inf and h'(0) > 0.
    % 
    % We take the real part only to be safe.
    y = min(real(roots([-sigma, alpha, gradnorm])));
    
    
    % Main Lanczos iteration
    gradnorm_reached = false;
    for j = 1 : min(options.maxiter_lanczos, n) - 1

        % Knowing that j Lanczos steps have been executed completely, now
        % execute the j+1st step to produce Q{j+1} and populate the
        % tridiagonal matrix T over the whole principal submatrix of size
        % j+1. This involves one Hessian call.
        %
        % In effect, we are computing this one step ahead. The reason is
        % that this makes it cheaper to compute the norm of the gradient of
        % the model, which is needed to check the stopping criterion (see
        % below).
        beta = rnorm(Hq_perp);
        % TODO: Figure out a sensible relative threshold
        if beta > 1e-12
            q = M.lincomb(x, 1/beta, Hq_perp);
        else
            % It appears the Krylov space maxed out (Hq is very nearly in
            % the space spanned by the existing Lanczos vectors). In order
            % to continue, the standard procedure is to generate a random
            % vector, and to orthogonalize it against the current basis.
            % This event is supposed to be rare.
            v = M.randvec(x);
            % Orthogonalize in the style of a modified Gram-Schmidt.
            for k = 1 : j
                v = M.lincomb(x, 1, v, -inner(v, Q{k}), Q{k});
            end
            q = M.lincomb(x, 1/rnorm(v), v);
        end
        Hq = Hess(q);
        hesscalls = hesscalls + 1;
        Hqm = M.lincomb(x, 1, Hq, -beta, Q{j});
        % In exact arithmetic, alpha = <Hq, q>. Doing the computations in
        % this order amounts to a modified GS, which may help numerically.
        alpha = inner(Hqm, q);
        Hq_perp = M.lincomb(x, 1, Hqm, -alpha, q);
        Q{j+1} = q;
        T(j, j+1) = beta;     %#ok<SPRIX>
        T(j+1, j) = beta;     %#ok<SPRIX>
        T(j+1, j+1) = alpha;  %#ok<SPRIX>
        % End of the Lanczos procedure for step j.

        % Computing the norm of the gradient of the model at the computed
        % step 'Qy' (linear combination of the Q's with coefficients y.)
        % We actually compute the norm of a vector of coordinates for the
        % gradient of the model in the basis Q{1}, ..., Q{j+1}.
        model_gradnorm = norm(gradnorm*eye(j+1, 1) + ...
                              T(1:j+1, 1:j)*y + ...
                              sigma*norm(y)*[y ; 0]);

        if options.verbosity >= 4
            fprintf('\nModel grad norm %.16e', model_gradnorm);
        end
        
        % Check termination condition
        if model_gradnorm <= options.theta*norm(y)^2
            stop_str = 'Model grad norm condition satisfied';
            gradnorm_reached = true;
            break;
        end
        
        % Minimize the cubic model restricted to the subspace spanned by
        % the available Lanczos vectors. In its current form, this solver
        % cannot reuse prior work from earlier Lanczos iterations: this may
        % be a point to improve.
        [y, newton_iter] = minimize_cubic_newton(T(1:j+1, 1:j+1), ...
                                     gradnorm*eye(j+1, 1), sigma, options);
        newton_iterations(j+1) = newton_iter;
        
    end
    
    % Check why we stopped iterating
    if ~gradnorm_reached
        stop_str = sprintf(['Reached max number of Lanczos iterations ' ...
               '(options.maxiter_lanczos = %d)'], options.maxiter_lanczos);
    end
    
    % Construct the tangent vector eta as a linear combination of the basis
    % vectors and make sure the result is tangent up to numerical accuracy.
    eta = lincomb(M, x, Q(1:numel(y)), y);
    eta = tangent(eta);
    % We could easily return the norm of eta as the norm of the coefficient
    % vector y here, but numerical errors might accumulate.
    
    % In principle we could avoid this call by computing an appropriate
    % linear combination of available vectors. For now at least, we favor
    % this numerically safer approach.
    Heta = Hess(eta);
    hesscalls = hesscalls + 1;
    
    stats = struct('newton_iterations', newton_iterations(1:numel(y)));
    
    if options.verbosity >= 4
        fprintf('\n');
    end
        
end
