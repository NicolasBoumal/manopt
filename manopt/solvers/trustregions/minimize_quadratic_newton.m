function [y, iter, lambda, status] = minimize_quadratic_newton(H, g, Delta, options)
% Minimize a quadratic via Newton root finding.
%
% [y, iter, lambda, status] = minimize_quadratic_newton(H, g, Delta, options)
%
% Inputs: a symmetric matrix H of size n, a nonzero vector g of length n,
% and an options structure. The code expects H to be tridiagonal, stored 
% as a sparse matrix.
%
% The main output is a vector y of length n, which should minimize
%
%   f(y) = g'*y + (1/2)*y'*H*y.
%   subject to y'*y <= Delta^2
%
% This is achieved by reducing the problem to a univariate root finding
% problem, where the unknown is a scalar lambda. This root is computed
% using a Newton method similar to the More-Sorensen algorithm.
%
% Other outputs are iter (the number of Newton iterations completed),
% lambda (a real scalar, see below) and status. The latter is 0 if the
% target tolerance was reached, 1 if subsequent iterations induce no
% significant change, and -1 if the algorithm return because it reached
% the maximum number of iterations (see the options structure.)
% Non-negative status values are considered successes.
%
% The options structure must contain the following fields (between
% parentheses are some recommended values):
%   options.verbosity (3): to control how much information this function
%   prints to the command window. Anything below 6 silences the function.
%   options.maxiter_newton (100): maximum number of Newton iterations.
%   options.tol_newton (1e-16): tolerance on the root finding accuracy. See
%   in code for details.
%
% The code is based on Algorithm 7.3.6 in
% More and Sorensen, "Computing a Trust Region Step", SIAM, 1983.
% https://epubs.siam.org/doi/10.1137/0904038
% 
% Lemma 2.1 and 2.3 in the referenced paper states y is optimal if and only
% if it there exists a real lambda such that
% 
% (H + lambda*I)y = -g,  lambda(Delta-||y||) = 0  and  H + lambda*I is psd,
% 
% where psd means positive semidefinite. The other way around, if we
% find the corresponding scalar lambda, than we can recover y by
% solving a linear system (though this system might not have a unique
% solution in full generality.) Thus, the general strategy is to search
% for lambda rather than for y.
%
% This code can be replaced with algorihm 5.2 as described in the GLTR 
% paper to speed up GLTR (see trs_lanczos.m for a link).
%
% See also: trustregions trs_lanczos

% This file is part of Manopt: www.manopt.org.
% Original authors: Victor Liao. January 20, 2023, with code adapted from
%    Naman Agarwal, Brian Bullins, Nicolas Boumal and Coralia Cartis.
% Contributors:
% Change log:
    
    n = size(H, 1);
    
    % Pick an initial lambda that is cheap to compute and that surely makes
    % the shifted H positive definite.
    lambda = norm(H, 1) + 2;
    H_shifted = H + lambda*speye(n);
    
    % Compute the smallest eigenvalue of H, as we know the target lambda
    % must be at least as large as the negative of that, so that the
    % shifted H will be positive semidefinite.
    % 
    % Since H ought to be sparse and tridiagonal, and since we only need
    % its smallest eigenvalue, this computation could be sped up
    % significantly. It does not appear to be a bottleneck, and eig is
    % simple and reliable, so we keep this for now.
    lambda_min = min(eig(H));
    left_barrier = max(0, -lambda_min);
    
    % Counter 'iter' holds the number of fully executed Newton iterations.
    iter = 0;
    while true
        
        if iter >= options.maxiter_newton
            % Iterations exceeded maximum number allowed.
            if options.verbosity >= 6
                fprintf(['lambda %.12e, ||y|| %.12e, lambda/delta %.12e, ' ...
                         'phi %.12e\n\n'], lambda, ynorm, lambda / Delta, phi);
            end
            status = -1;
            return;
        end
        % If lambda has the correct value and the shifted H is positive
        % definite, then this y is a minimizer.
        y = -(H_shifted\g);
        ynorm = norm(y);

        % If the following quantity is zero, we have found a solution.
        phi = 1/Delta - 1/ynorm;
        
        % Check if it is close enough to zero to stop.
        if abs(phi) <= options.tol_newton*ynorm
            if options.verbosity >= 6
                fprintf(['lambda %.12e, ||y|| %.12e, lambda/delta %.12e, ' ...
                         'phi %.12e\n\n'], lambda, ynorm, lambda / Delta, phi);
            end
            status = 0;
            return;
        end

        R = chol(H_shifted);
        q = (R.')\y;
        qnorm = norm(q);
        del_lambda = (ynorm/qnorm)^2 * (ynorm - Delta)/Delta;
        iter = iter + 1;

        % If the Newton step would bring us left of the left barrier, jump
        % instead to the midpoint between the left barrier and the current
        % lambda.
        if lambda + del_lambda <= left_barrier
            del_lambda = -.5*(lambda - left_barrier);
        end

        % If the step is so small that it numerically does not make a
        % difference when added to the current lambda, we stop.
        if abs(del_lambda) <= eps(lambda)
            if options.verbosity >= 6
                fprintf(['lambda %.12e, ||y|| %.12e, lambda/delta %.12e, ' ...
                         'phi %.12e\n\n'], lambda, ynorm, lambda / Delta, phi);
            end
            status = 1;
            return;
        end

        % Update lambda
        H_shifted = H_shifted + del_lambda*speye(n);
        lambda = lambda + del_lambda;
        
        
        if options.verbosity >= 6
            fprintf(['lambda %.12e, ||y|| %.12e, lambda/delta %.12e, ' ...
                     'phi %.12e\n\n'], lambda, ynorm, lambda / Delta, phi);
        end

    end

end
