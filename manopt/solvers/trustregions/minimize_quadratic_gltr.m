function [y, iter, lambda, status] = minimize_quadratic_gltr(H, g, Delta, options)
    n = size(H, 1);

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
    
    
    % Pick an initial lambda that is cheap to compute and that makes the 
    % shifted H positive definite according to the paper
    lambda = (1 - lambda_min);
    H_shifted = H + lambda*speye(n);
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

        try
            R = chol(H_shifted);
        catch ME
            disp(H_shifted);
            disp(lambda_min);
            disp(lambda);
            [h2, limitedbyTR, ~] = TRSgep(H, g, Delta);
        end
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
