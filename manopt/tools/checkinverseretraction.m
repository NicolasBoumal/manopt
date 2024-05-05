function checkinverseretraction(M, x, v)
% Check the order of agreement of an inverse retraction with the log.
% 
% function checkinverseretraction(M)
% function checkinverseretraction(M, x)
% function checkinverseretraction(M, x, v)
%
% checkinverseretraction performs a numerical test to check the order of
% agreement between the inverse retraction and the logarithmic map in a
% given Manopt manifold structure M. The test is performed at the point x
% if it is provided (otherwise, the point is picked at random) and along
% the tangent vector v at x if one is provided (otherwise, a tangent vector
% at x is picked at random.)
%
% See also: checkretraction checkmanifold checkdiff checkgradient checkhessian

% This file is part of Manopt: www.manopt.org.
% Original author: Ronny Bergmann, May 3rd, 2024.
% Contributors: 
% Change log: 

    if ~isfield(M, 'exp')
        error(['This manifold has no exponential (M.exp) which is ' ...
               'required to generate points at a certain distance from x.']);
    end
    if ~isfield(M, 'log')
        error(['This manifold has no logarithm (M.log): ' ...
               'no reference to compare the inverse retraction.']);
    end
    if ~isfield(M, 'norm')
        error(['This manifold has no norm (M.norm): ' ...
               'this is required to run this check.']);
    end

    if ~exist('x', 'var') || isempty(x)
        x = M.rand();
        v = M.randvec(x);
    end
    
    if ~exist('v', 'var') || isempty(v)
        v = M.randvec(x);
    end
    
    % Compare the retraction and the exponential over steps of varying
    % length, on a wide log-scale.
    tt = logspace(-12, 0, 251);
    ee = zeros(size(tt));
    for k = 1 : numel(tt)
        t = tt(k);
        y = M.exp(M, v, t);
        diff = M.lincomb(x, 1, M.log(x, y), -1, M.invretr(x, y));
        ee(k) = M.norm(x, diff);
    end
    
    % Plot the difference between the exponential and the retration over
    % that span of steps, in log-log scale.
    loglog(tt, ee);
    
    % Include visual references of slope 2 and 3.
    % Slope 3
    line('xdata', [1e-12 1e0], 'ydata', [1e-30 1e6], ...
         'color', 'k', 'LineStyle', '--', ...
         'YLimInclude', 'off', 'XLimInclude', 'off');
    % Slope 2
    line('xdata', [1e-14 1e0], 'ydata', [1e-20 1e8], ...
         'color', 'k', 'LineStyle', ':', ...
         'YLimInclude', 'off', 'XLimInclude', 'off');
     

    % Figure out the slope of the error in log-log, by identifying a piece
    % of the error curve which is mostly linear.
    window_len = 10;
    [range, poly] = identify_linear_piece(log10(tt), log10(ee), window_len);
    hold all;
    loglog(tt(range), 10.^polyval(poly, log10(tt(range))), 'LineWidth', 3);
    hold off;
    
    xlabel('Step size multiplier t');
    ylabel('Distance between M.log(x, y) and M.invretr(x, y) for y = M.exp(x, v, t)');
    title(sprintf(['Inverse retraction check.\n' ...
                   'A slope of 2 is required, 3 is desired.\n' ...
                   'Also read text output in command prompt.']));
    
    fprintf('Check agreement between M.log and M.invretr. Please check the\n');
    fprintf('factory file of M to ensure M.log is a proper logarithm.\n');
    fprintf('The slope must be at least 2 to have a proper invese retraction.\n');
    fprintf('For the inverse retraction to be second order, the slope should be 3.\n');
    fprintf('It appears the slope is: %g.\n', poly(1));
    fprintf(['Note: If M.log and M.invretr are identical, ' ...
             'the following is about zero: %g.\n'], norm(ee));
    fprintf(['      If so, the inverse retraction is fine and ' ...
             'the slope test is irrelevant.\n']);
end
