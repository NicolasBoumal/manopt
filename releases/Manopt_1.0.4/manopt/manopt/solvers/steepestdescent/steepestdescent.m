function [x cost info] = steepestdescent(problem, x, options)
% Steepest descent (gradient descent) minimization algorithm for Manopt.
%
% function [x cost info] = steepestdescent(problem)
% function [x cost info] = steepestdescent(problem, x0)
% function [x cost info] = steepestdescent(problem, x0, options)
% function [x cost info] = steepestdescent(problem, [], options)
%
% Apply the steepest descent minimization algorithm to the problem defined
% in the problem structure, starting at x0 if it is provided (otherwise, at
% a random point on the manifold). To specify options whilst not specifying
% an initial guess, give x0 as [] (the empty matrix).
%
% None of the options are mandatory. See the documentation for details.
%
% Stopping criteria (options fields):
% minstepsize : stop if the line search asks to retract along a vector of
%               length less than minstepsize.
% maxtime, maxiter, tolgradnorm... : standard (see Manopt documentation).
%
% Line search algorithm: a default line search algorithm is used. To use a
% different one, set the options.linesearch field to be a function handle
% for a valid line search function.
% See manopt/solvers/linesearch/linesearch for an example.
% An alternate built-in linesearch can be used with:
% options.linesearch = @linesearch_adaptive;
%
% The outputs are the standard outputs for solvers in Manopt:
% x : the best point reached;
% cost : the cost at the returned x;
% info : a struct-array with information logged along the iterations.
%
% See also: linesearch conjugategradient

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 

    
    % Verify that the problem description is sufficient for the solver.
    if ~canGetCost(problem)
        warning('manopt:getCost', ...
                'No cost provided. The algorithm will likely abort.');  
    end
    if ~canGetGradient(problem)
        warning('manopt:getGradient', ...
                'No gradient provided. The algorithm will likely abort.');    
    end

    % Set local defaults here
    localdefaults.minstepsize = 1e-10;
    localdefaults.maxiter = 1000;
    localdefaults.tolgradnorm = 1e-6;
    localdefaults.linesearch = @linesearch;
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    % Create a store database
    storedb = struct();
    
    timetic = tic();
    
    % If no initial point x is given by the user, generate one at random.
    if ~exist('x', 'var') || isempty(x)
        x = problem.M.rand();
    end
    
    % Compute objective-related quantities for x
    [cost grad storedb] = getCostGrad(problem, x, storedb);
    gradnorm = problem.M.norm(x, grad);
    
    % Iteration counter (at any point, iter is the number of fully executed
    % iterations so far)
    iter = 0;
    
    % Save stats in a struct array info, and preallocate
    % (see http://people.csail.mit.edu/jskelly/blog/?x=entry:entry091030-033941)
    stats = savestats();
    info(1) = stats;
    info(min(10000, options.maxiter+1)).iter = [];
    
    % Initial line search memory
    lsmem = [];
    
    if options.verbosity >= 2
        fprintf(' iter\t    cost val\t grad. norm\n');
    end
    
    % Start iterating until stopping criterion triggers
    while true

        % Display iteration information
        if options.verbosity >= 2
            fprintf('%5d\t%+.4e\t%.4e\n', iter, cost, gradnorm);
        end
        
        % Start timing this iteration
        timetic = tic();
        
        % Run standard stopping criterion checks
        [stop reason] = stoppingcriterion(problem, x, options, ...
                                                             info, iter+1);
        
        % Run specific stopping criterion check
        if ~stop && stats.stepsize < options.minstepsize
            stop = true;
            reason = 'Last stepsize smaller than minimum allowed.';
        end
    
        if stop
            if options.verbosity >= 1
                fprintf([reason '\n']);
            end
            break;
        end

        % Compute a normalized descent direction
        desc_dir = problem.M.lincomb(x, -1/gradnorm, grad);
        
        % Execute line search
        [stepsize newx storedb lsmem lsstats] = options.linesearch(...
           problem, x, desc_dir, cost, -gradnorm, options, storedb, lsmem);
        
        % Compute the new objective-related quantities for x
        [newcost newgrad storedb] = getCostGrad(problem, newx, storedb);
        newgradnorm = problem.M.norm(newx, newgrad);
        
        % Make sure we don't use too much memory for the store database
        storedb = purgeStoredb(storedb, options.storedepth);
        
        % Update iterate info
        x = newx;
        cost = newcost;
        grad = newgrad;
        gradnorm = newgradnorm;
        
        % iter is the number of iterations we have accomplished.
        iter = iter + 1;
        
        % Log statistics for freshly executed iteration
        stats = savestats();
        info(iter+1) = stats; %#ok<AGROW>
        
    end
    
    
    info = info(1:iter+1);

    if options.verbosity >= 1
        fprintf('Total time is %f [s] (excludes statsfun)\n', ...
                                                           info(end).time);
    end
    
    
    
    % Routine in charge of collecting the current iteration stats
    function stats = savestats()
        stats.iter = iter;
        stats.cost = cost;
        stats.gradnorm = gradnorm;
        if iter == 0
            stats.stepsize = nan;
            stats.time = toc(timetic);
            stats.linesearch = [];
        else
            stats.stepsize = stepsize;
            stats.time = info(iter).time + toc(timetic);
            stats.linesearch = lsstats;
        end
        stats = applyStatsfun(problem, x, storedb, options, stats);
    end
    
end
