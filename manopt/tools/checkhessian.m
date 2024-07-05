function checkhessian(problem, x, d)
% Checks the consistency of the cost function and the Hessian.
%
% function checkhessian(problem)
% function checkhessian(problem, x)
% function checkhessian(problem, x, d)
%
% checkhessian performs a numerical test to check that the gradient and
% Hessian defined in the problem structure agree up to second order with
% the cost function at some point x, along some direction d.
% The test is based on a truncated Taylor series (see online Manopt
% documentation).
% 
% It is also tested that the result of applying the Hessian along that
% direction is indeed a tangent vector, and that the Hessian operator is
% symmetric w.r.t. the Riemannian metric.
% 
% Both x and d are optional and will be sampled at random if omitted.
%
% The slope test requires the exponential map of the manifold, or at least
% a second-order retraction. If neither M.exp nor M.retr2 are available,
% the retraction M.retr is used and a message is issued to instruct the
% user to check whether M.retr actually is second-order or not. If it is
% not, the slope test is only valid at critical points of the cost function
% (which can be computed by optimization for example.)
%
% See also: checkdiff checkgradient checkretraction

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
% 
%   April 3, 2015 (NB):
%       Works with the new StoreDB class system.
%
%   Nov. 1, 2016 (NB):
%       Issues a call to getGradient rather than getDirectionalDerivative.
%
%   March 26, 2017 (JB):
%       Detects if the approximated quadratic model is exact
%       and provides the user with the corresponding feedback.
%
%   Dec. 6, 2017 (NB):
%       Added message in case tangent2ambient might be necessary in
%       defining ehess (this was a common difficulty for users.)
%
%   Aug. 2, 2018 (NB):
%       Using storedb.remove() to avoid unnecessary cache build-up.
%
%   Sep. 6, 2018 (NB):
%       Now checks whether M.exp() is available; uses retraction otherwise
%       and issues a message that the user should check whether the
%       retraction is second-order or not.
%
%   Feb. 1, 2020 (NB):
%       Added an explicit linearity check.
%
%   July 2, 2024 (NB):
%       Use exp, or retr2, or retr (in that order). Only issue a slope
%       warning if using retr (because retr2 is second order). Also using
%       the new tool offtangent to check that tangent vectors are tangent.
%       Also styled the output with bold.

    fprintf('<strong># Hessian check</strong>\n');
        
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
    
    %% Check that the gradient at x and the Hessian at x along d
    %% yield a second order model of the cost function.
    
    % Compute the value f0 at f, directional derivative df0 at x along d
    % based on the gradient at x, and Hessian along [d, d].
    storedb = StoreDB();
    xkey = storedb.getNewKey();
    f0 = getCost(problem, x, storedb, xkey);
    gradx = getGradient(problem, x, storedb, xkey);
    df0 = problem.M.inner(x, d, gradx);
    hessxd = getHessian(problem, x, d, storedb, xkey);
    d2f0 = problem.M.inner(x, d, hessxd);
    
    
    % Pick a stepping function. Order of preference: exp > retr2 > retr.
    stepper = problem.M.retr;
    stepper_is_secondorder = false;  % retr may be second order, but unsure
    if isfield(problem.M, 'retr2')
        stepper = problem.M.retr2;
        stepper_is_secondorder = true;
    end
    if isfield(problem.M, 'exp')
        stepper = problem.M.exp;
        stepper_is_secondorder = true;
    end
    if stepper_is_secondorder
        extra_message = '';
    else
        fprintf( ...
            ['* M.exp and M.retr2 are not available: using M.retr.\n' ...
             '* Please check the manifold documentation to see if\n' ...
             '* the retraction is second order. If not, the slope\n' ...
             '* test is allowed to fail at non-critical x.\n']);
        extra_message = ...
            ['(But do mind the message above: the slope may ' ...
             'be\nallowed to be 2 at non-critical points x.)\n'];
    end
    
    
    % Compute the value of f at points on the geodesic (or approximation
    % of it) originating from x, along direction d, for stepsizes in a
    % large range given by t.
    t = logspace(-8, 0, 51);
    value = zeros(size(t));
    for i = 1 : length(t)
        y = stepper(x, d, t(i));
        ykey = storedb.getNewKey();
        value(i) = getCost(problem, y, storedb, ykey);
        storedb.remove(ykey); % no need to keep it in memory
    end
    
    % Compute the quadratic approximation of the cost function using f0,
    % df0 and d2f0 at the same points.
    model = polyval([.5*d2f0, df0, f0], t);
    
    % Compute the approximation error
    err = abs(model - value);
    
    % And plot it.
    loglog(t, err);
    title(sprintf(['Hessian check.\nThe slope of the continuous line ' ...
                   'should match that of the dashed\n(reference) line ' ...
                   'over at least a few orders of magnitude for t.']));
    xlabel('t');
    ylabel('Approximation error');
    
    line('xdata', [1e-8, 1e0], 'ydata', [1e-16, 1e8], ...
         'color', 'k', 'LineStyle', '--', ...
         'YLimInclude', 'off', 'XLimInclude', 'off');
    
    
    if ~all( err < 1e-12 )
        % In a numerically reasonable neighborhood, the error should
        % decrease as the cube of the stepsize, i.e., in loglog scale, the
        % error should have a slope of 3.
        isModelExact = false;
        window_len = 10;
        [range, poly] = identify_linear_piece(log10(t), log10(err), window_len);
    else
        % The 2nd order model is exact: all errors are (numerically) zero
        % Fit line from all points, use log scale only in t.
        isModelExact = true;
        range = 1:numel(t);
        poly = polyfit(log10(t), err, 1);
        % Set mean error in log scale for plot
        poly(end) = log10(poly(end));
        % Change title to something more descriptive for this special case.
        title(sprintf(...
              ['Hessian check.\n'...
               'It seems the quadratic model is exact:\n'...
               'Model error is numerically zero for all t.']));
    end
    hold all;
    loglog(t(range), 10.^polyval(poly, log10(t(range))), 'LineWidth', 3);
    hold off;
    
    if ~isModelExact
        fprintf(['The slope should be 3. It appears to be: ' ...
                 '<strong>%g</strong>.\n'], poly(1));
        fprintf(['If it is far from 3, then the gradient or the ' ...
                 'Hessian might be erroneous.\n', ...
                 extra_message]);
    else
        fprintf(['The quadratic model appears to be exact ' ...
                 '(within numerical precision),\n'...
                 'hence the slope computation is irrelevant.\n']);
    end

    
    %% Check that the Hessian at x along direction d is a tangent vector.
    hess = getHessian(problem, x, d, storedb, xkey);
    err = offtangent(problem.M, x, hess);
    if ~isnan(err)
        fprintf('Hess f(x)[d] must be a tangent vector at x.\n');
        fprintf(['If so, the following number is zero up to machine ' ...
                 'precision: <strong>%g</strong>.\n'], err);
        if ~isinf(err)
            fprintf(['If it is far from 0, the Hessian returns ' ...
                     'non-tangent vectors.\n']);
        else
            fprintf(['The output is Inf, suggesting Hess f(x)[d] is ' ...
                     'not in the right format.\nCheck array sizes.']);
        end
    else
        fprintf(['Unfortunately, Manopt was unable to verify that the '...
                 'output of the Hessian call is indeed a tangent ' ...
                 'vector.\nPlease verify this manually or implement ' ...
                 'the ''tangent'' or the ''offtangent'' function in ' ...
                 'your manifold structure.']);
    end    
    
    %% Check that the Hessian at x is linear and symmetric.
    d1 = problem.M.randvec(x);
    d2 = problem.M.randvec(x);
    h1 = getHessian(problem, x, d1, storedb, xkey);
    h2 = getHessian(problem, x, d2, storedb, xkey);
    
    % Linearity check
    a = randn(1);
    b = randn(1);
    ad1pbd2 = problem.M.lincomb(x, a, d1, b, d2);
    had1pbd2 = getHessian(problem, x, ad1pbd2, storedb, xkey);
    ahd1pbhd2 = problem.M.lincomb(x, a, h1, b, h2);
    errvec = problem.M.lincomb(x, 1, had1pbd2, -1, ahd1pbhd2);
    errvecnrm = problem.M.norm(x, errvec);
    had1pbd2nrm = problem.M.norm(x, had1pbd2);
    fprintf('The Hessian at x must be linear on the tangent space at x.\n');
    fprintf(['If so, ||a*H[d1] + b*H[d2] - H[a*d1+b*d2]|| is zero up to ' ...
             'machine precision.\n\tValue: <strong>%g</strong> ' ...
             '(norm of H[a*d1+b*d2]: %g)\n'], ...
             errvecnrm, had1pbd2nrm);
    fprintf('If it is far from 0, then the Hessian is not linear.\n');
    
    % Symmetry check
    v1 = problem.M.inner(x, d1, h2);
    v2 = problem.M.inner(x, h1, d2);
    value = v1-v2;
    fprintf('The Hessian at x must be symmetric on the tangent space at x.\n');
    fprintf(['If so, <d1, H[d2]> - <H[d1], d2> is zero up to machine precision.' ...
             '\n\tValue: %g - %g = <strong>%g</strong>.\n'], v1, v2, value);
    fprintf('If it is far from 0, then the Hessian is not symmetric.\n');
    
    %% Check if the manifold at hand is one of those for which there should
    %  be a call to M.tangent2ambient, as this is a common mistake. If so,
    %  issue an additional message. Ideally, one would just need to check
    %  for the presence of tangent2ambient, but productmanifold (for
    %  example) generates one of those in all cases, even if it is just an
    %  identity map.
    if isfield(problem.M, 'tangent2ambient_is_identity') && ...
                                     ~problem.M.tangent2ambient_is_identity
        
        fprintf('\n\n!! <strong>Special message for this manifold</strong>\n');
        fprintf('\nIf all above looks good, great. If not, read on.\n\n');
        fprintf(['For this manifold, tangent vectors are represented\n' ...
                 'differently from their ambient space representation.\n' ...
                 'In practice, this means that when defining\n' ...
                 'v = problem.ehess(x, u), one may need to call\n' ...
                 'u = problem.M.tangent2ambient(x, u) first, so as to\n'...
                 'transform u into an ambient vector, if this is more\n' ...
                 'convenient. The output of ehess should be an ambient\n' ...
                 'vector (it will be transformed to a tangent vector\n' ...
                 'automatically).\n\nFor more on this concept, see ' ...
                 '<a href="https://www.manopt.org/rotationsfactory.html">' ...
                 'the webpage about rotationsfactory</a>.\n\n']);
        
    end

    if ~canGetHessian(problem)
        norm_grad = problem.M.norm(x, gradx);
        fprintf(['\nWhen using checkhessian with a finite difference ' ...
                 'approximation, the norm of the residual\nshould be ' ...
                 'compared against the norm of the gradient at the ' ...
                 'point under consideration (%.3e).\nFurthermore, it ' ...
                 'is expected that the FD operator is only approximately' ...
                 ' symmetric.\nOf course, the slope can also be off.\n'], ...
                 norm_grad);
    end
    
end
