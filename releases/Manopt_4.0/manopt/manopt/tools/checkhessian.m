function checkhessian(problem, x, d)
% Checks the consistency of the cost function and the Hessian.
%
% function checkhessian(problem)
% function checkhessian(problem, x)
% function checkhessian(problem, x, d)
%
% checkhessian performs a numerical test to check that the directional
% derivatives and Hessian defined in the problem structure agree up to
% second order with the cost function at some point x, along some direction
% d. The test is based on a truncated Taylor series (see online Manopt
% documentation).
% 
% It is also tested that the result of applying the Hessian along that
% direction is indeed a tangent vector, and that the Hessian operator is
% symmetric w.r.t. the Riemannian metric.
% 
% Both x and d are optional and will be sampled at random if omitted.
%
% See also: checkdiff checkgradient checkretraction

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
%
%   March 26, 2017 (JB):
%       Detects if the approximated quadratic model is exact
%       and provides the user with the corresponding feedback.
% 
%   April 3, 2015 (NB):
%       Works with the new StoreDB class system.
%
%   Nov. 1, 2016 (NB):
%       Issues a call to getGradient rather than getDirectionalDerivative.

        
    % Verify that the problem description is sufficient.
    if ~canGetCost(problem)
        error('It seems no cost was provided.');
    end
    if ~canGetGradient(problem)
        warning('manopt:checkhessian:nograd', ...
                'It seems no gradient was provided.');
    end
    if ~canGetHessian(problem)
        warning('manopt:checkhessian:nohess', ...
                'It seems no Hessian was provided.');
    end
    
    x_isprovided = exist('x', 'var') && ~isempty(x);
    d_isprovided = exist('d', 'var') && ~isempty(d);
    
    if ~x_isprovided && d_isprovided
        error('If d is provided, x must be too, since d is tangent at x.');
    end
    
    % If x and / or d are not specified, pick them at random.
    if ~x_isprovided
        x = problem.M.rand();
    end
    if ~d_isprovided
        d = problem.M.randvec(x);
    end
    
    %% Check that the directional derivative and the Hessian at x along d
    %% yield a second order model of the cost function.
    
    % Compute the value f0 at f, directional derivative df0 at x along d,
    % and Hessian along [d, d].
    storedb = StoreDB();
    xkey = storedb.getNewKey();
    f0 = getCost(problem, x, storedb, xkey);
    df0 = problem.M.inner(x, d, getGradient(problem, x, storedb, xkey));
    d2f0 = problem.M.inner(x, d, getHessian(problem, x, d, storedb, xkey));
    
    % Compute the value of f at points on the geodesic (or approximation
    % of it) originating from x, along direction d, for stepsizes in a
    % large range given by h.
    h = logspace(-8, 0, 51);
    value = zeros(size(h));
    for i = 1 : length(h)
        y = problem.M.exp(x, d, h(i));
        ykey = storedb.getNewKey();
        value(i) = getCost(problem, y, storedb, ykey);
    end
    
    % Compute the quadratic approximation of the cost function using f0,
    % df0 and d2f0 at the same points.
    model = polyval([.5*d2f0 df0 f0], h);
    
    % Compute the approximation error
    err = abs(model - value);
    
    % And plot it.
    loglog(h, err);
    title(sprintf(['Hessian check.\nThe slope of the continuous line ' ...
                   'should match that of the dashed\n(reference) line ' ...
                   'over at least a few orders of magnitude for h.']));
    xlabel('h');
    ylabel('Approximation error');
    
    line('xdata', [1e-8 1e0], 'ydata', [1e-16 1e8], ...
         'color', 'k', 'LineStyle', '--', ...
         'YLimInclude', 'off', 'XLimInclude', 'off');
    
    
    if ~all( err < 1e-12 )
        % In a numerically reasonable neighborhood, the error should
        % decrease as the cube of the stepsize, i.e., in loglog scale, the
        % error should have a slope of 3.
        isModelExact = false;
        window_len = 10;
        [range, poly] = identify_linear_piece(log10(h), log10(err), window_len);
    else
        % The 2nd order model is exact: all errors are (numerically) zero
        % Fit line from all points, use log scale only in h.
        isModelExact = true;
        range = 1:numel(h);
        poly = polyfit(log10(h), err, 1);
        % Set mean error in log scale for plot
        poly(end) = log10(poly(end));
        % Change title to something more descriptive for this special case.
        title(sprintf(...
              ['Hessian check.\n'...
               'It seems the quadratic model is exact:\n'...
               'Model error is numerically zero for all h.']));
    end
    hold all;
    loglog(h(range), 10.^polyval(poly, log10(h(range))), 'LineWidth', 3);
    hold off;
    
    if ~isModelExact
        fprintf('The slope should be 3. It appears to be: %g.\n', poly(1));
        fprintf(['If it is far from 3, then directional derivatives or ' ...
                 'the Hessian might be erroneous.\n']);
        fprintf(['Note: if the exponential map is only approximate, and it '...
                 'is not a second-order approximation,\nthen it is normal ' ...
                 'for the slope test to reach 2 instead of 3. Check the ' ...
                 'factory for this.\n' ...
                 'If tested at a critical point, then even for a first-order '...
                 'retraction the slope test should yield 3.\n']);
    else
        fprintf(['The quadratic model appears to be exact ' ...
                 '(within numerical precision),\n'...
                 'hence the slope computation is irrelevant.\n']);
    end

    
    %% Check that the Hessian at x along direction d is a tangent vector.
    if isfield(problem.M, 'tangent')
        hess = getHessian(problem, x, d, storedb, xkey);
        phess = problem.M.tangent(x, hess);
        residual = problem.M.lincomb(x, 1, hess, -1, phess);
        err = problem.M.norm(x, residual);
        fprintf('The residual should be zero, or very close. ');
        fprintf('Residual: %g.\n', err);
        fprintf(['If it is far from 0, then the Hessian is not in the ' ...
                 'tangent plane.\n']);
    else
        fprintf(['Unfortunately, Manopt was unable to verify that the '...
                 'Hessian is indeed a tangent vector.\nPlease verify ' ...
                 'this manually.']);
    end    
    
    %% Check that the Hessian at x is symmetric.
    d1 = problem.M.randvec(x);
    d2 = problem.M.randvec(x);
    h1 = getHessian(problem, x, d1, storedb, xkey);
    h2 = getHessian(problem, x, d2, storedb, xkey);
    v1 = problem.M.inner(x, d1, h2);
    v2 = problem.M.inner(x, h1, d2);
    value = v1-v2;
    fprintf(['<d1, H[d2]> - <H[d1], d2> should be zero, or very close.' ...
             '\n\tValue: %g - %g = %g.\n'], v1, v2, value);
    fprintf('If it is far from 0, then the Hessian is not symmetric.\n');
    
end
