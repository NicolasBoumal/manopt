function function_call_counters()
% Manopt example on how to count function calls during optimization.

% This file is part of Manopt and is copyrighted. See the license file.
%
% Main author: Nicolas Boumal, July 24, 2018
% Contributors:

    % Setup an optimization problem for testing.
    n = 1000;
    A = randn(n);
    A = .5*(A+A');
    
    manifold = spherefactory(n);
    problem.M = manifold;
    
    % These global counters are used to keep track of how many function
    % calls are issued to cost, gradient and Hessian.
    fcounter = 0;
    gcounter = 0;
    hcounter = 0;
    
    % Define the problem cost function and its gradient.
    problem.cost = @cost;
    function f = cost(x)
        f = -x'*(A*x);
        fcounter = fcounter + 1;
    end
    problem.egrad = @egrad;
    function g = egrad(x)
        g = -2*A*x;
        gcounter = gcounter + 1;
    end
    problem.ehess = @ehess;
    function h = ehess(x, xdot) %#ok<INUSL>
        h = -2*A*xdot;
        hcounter = hcounter + 1;
    end
    
    % Setup a callback to log statistics.
    % Note: inline definitions such as stats.costcalls = @(x) fcounter; do
    % not work because they do not access fcounter as a global variable;
    % rather, the value of fcounter at the time of definition is used and
    % then fixed.
    stats.costcalls = @fcount;
    function fc = fcount(x) %#ok<INUSD>
        fc = fcounter;
    end
    stats.gradcalls = @gcount;
    function gc = gcount(x) %#ok<INUSD>
        gc = gcounter;
    end
    stats.hesscalls = @hcount;
    function hc = hcount(x) %#ok<INUSD>
        hc = hcounter;
    end
    options.statsfun = statsfunhelper(stats);

    % Solve.
    [x, xcost, info] = trustregions(problem, [], options); %#ok<ASGLU>
    
    % Display some statistics.
    figure;
    subplot(2, 2, 1);
    semilogy([info.iter], [info.gradnorm], '.-');
    xlabel('Iteration #');
    ylabel('Gradient norm');
    subplot(2, 2, 2);
    semilogy([info.costcalls], [info.gradnorm], '.-');
    xlabel('# cost calls');
    ylabel('Gradient norm');
    subplot(2, 2, 3);
    semilogy([info.gradcalls], [info.gradnorm], '.-');
    xlabel('# gradient calls');
    ylabel('Gradient norm');
    subplot(2, 2, 4);
    semilogy([info.hesscalls], [info.gradnorm], '.-');
    xlabel('# Hessian calls');
    ylabel('Gradient norm');
    
end
