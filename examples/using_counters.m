function using_counters()
% Manopt example on how to use counters during optimization. Typical uses,
% as demonstrated here, include counting calls to cost, gradient and
% Hessian functions.
%
% See also: statscounters incrementcounter statsfunhelper

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, July 27, 2018.
% Contributors: 
% Change log: 

    % Setup an optimization problem to illustrate the use of counters
    n = 1000;
    A = randn(n);
    A = .5*(A+A');
    
    manifold = spherefactory(n);
    problem.M = manifold;
    
    % Define the problem cost function and its gradient.
    problem.cost = @cost;
    function [f, store] = cost(x, store)
        f = -x'*(A*x);
        % Increment a counter for the number of calls to the cost function
        % and one for the number of matrix-vector products involving A.
        % The names of the counters (costcalls and Aproducts) are for us to
        % choose: they only need to be valid structure field names. They
        % need not have been defined in advance.
        store = incrementcounter(store, 'costcalls');
        store = incrementcounter(store, 'Aproducts');
    end
    problem.egrad = @egrad;
    function [g, store] = egrad(x, store)
        g = -2*A*x;
        % Count the number of calls to the gradient function and also count
        % the matrix-vector product involving A. Notice here that, if we
        % used the store to cache the product A*x (which is also used in
        % the computation of the cost function), then we could spare a
        % product. We could comment out the incrementation of Aproducts to
        % investigate how much we could win by implementing proper caching
        % of A*x. Then, if it appears significant, 
        store = incrementcounter(store, 'gradcalls');
        store = incrementcounter(store, 'Aproducts');
    end
    problem.ehess = @ehess;
    function [h, store] = ehess(x, xdot, store) %#ok<INUSL>
        h = -2*A*xdot;
        % Count the number of calls to the Hessian operator and also count
        % the matrix-vector product with A (this one cannot be cached).
        store = incrementcounter(store, 'hesscalls');
        store = incrementcounter(store, 'Aproducts');
    end

    % General comment: counters are stored as doubles; they can harbor non
    % integer values if desired. The third argument to incrementcounter is
    % the amount by which to increment the counter (1 by default). Thus,
    % one could also time a portion of the code in the cost/grad/Hessian
    % code using tic/toc, and increment a dedicated counter with toc() as
    % third argument.
    
    % Setup a callback to log statistics. We use a combination of
    % statscounters and of statsfunhelper to indicate which counters we
    % want the optimization algorithm to log. Here, stats is a structure
    % where each field is a function handle corresponding to one of the
    % counters. Before passing stats to statsfunhelper, we could decide to
    % add more fields to stats to log other things as well.
    stats = statscounters({'costcalls', 'gradcalls', 'hesscalls', 'Aproducts'});
    options.statsfun = statsfunhelper(stats);

    % Solve with two solvers to compare.
    [x, xcost, infotr] = trustregions(problem, [], options); %#ok<ASGLU>
    [x, xcost, infocg] = conjugategradient(problem, [], options); %#ok<ASGLU>
    
    % Display some statistics. The logged data is available in the info
    % struct-arrays. Notice how the counters are available by their
    % corresponding field name.
    figure(1);
    subplot(2, 3, 1);
    semilogy([infotr.iter], [infotr.gradnorm], '.-', ...
             [infocg.iter], [infocg.gradnorm], '.-');
    legend('RTR', 'RCG');
    xlabel('Iteration #');
    ylabel('Gradient norm');
    subplot(2, 3, 2);
    semilogy([infotr.costcalls], [infotr.gradnorm], '.-', ...
             [infocg.costcalls], [infocg.gradnorm], '.-');
    xlabel('# cost calls');
    ylabel('Gradient norm');
    subplot(2, 3, 3);
    semilogy([infotr.gradcalls], [infotr.gradnorm], '.-', ...
             [infocg.gradcalls], [infocg.gradnorm], '.-');
    xlabel('# gradient calls');
    ylabel('Gradient norm');
    subplot(2, 3, 4);
    semilogy([infotr.hesscalls], [infotr.gradnorm], '.-', ...
             [infocg.hesscalls], [infocg.gradnorm], '.-');
    xlabel('# Hessian calls');
    ylabel('Gradient norm');
    subplot(2, 3, 5);
    semilogy([infotr.Aproducts], [infotr.gradnorm], '.-', ...
             [infocg.Aproducts], [infocg.gradnorm], '.-');
    xlabel('# matrix-vector products');
    ylabel('Gradient norm');
    subplot(2, 3, 6);
    semilogy([infotr.time], [infotr.gradnorm], '.-', ...
             [infocg.time], [infocg.gradnorm], '.-');
    xlabel('Computation time [s]');
    ylabel('Gradient norm');
    
end
