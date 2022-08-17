function [x, cost, info, options] = steepestdescent(problem, x, options)
% Steepest descent (gradient descent) minimization algorithm for Manopt.
%
% function [x, cost, info, options] = steepestdescent(problem)
% function [x, cost, info, options] = steepestdescent(problem, x0)
% function [x, cost, info, options] = steepestdescent(problem, x0, options)
% function [x, cost, info, options] = steepestdescent(problem, [], options)
%
% Apply the steepest descent minimization algorithm to the problem defined
% in the problem structure, starting at x0 if it is provided (otherwise, at
% a random point on the manifold). To specify options whilst not specifying
% an initial guess, give x0 as [] (the empty matrix).
%
% In most of the examples bundled with the toolbox (see link below), the
% solver can be replaced by the present one if need be.
%
% The outputs x and cost are the best reached point on the manifold and its
% cost. The struct-array info contains information about the iterations:
%   iter : the iteration number (0 for the initial guess)
%   cost : cost value
%   time : elapsed time in seconds
%   gradnorm : Riemannian norm of the gradient
%   stepsize : norm of the last tangent vector retracted
%   linesearch : information logged by options.linesearch
%   And possibly additional information logged by options.statsfun.
% For example, type [info.gradnorm] to obtain a vector of the successive
% gradient norms reached.
%
% The options structure is used to overwrite the default values. All
% options have a default value and are hence optional. To force an option
% value, pass an options structure with a field options.optionname, where
% optionname is one of the following and the default value is indicated
% between parentheses:
%
%   tolgradnorm (1e-6)
%       The algorithm terminates if the norm of the gradient drops below this.
%   maxiter (1000)
%       The algorithm terminates if maxiter iterations have been executed.
%   maxtime (Inf)
%       The algorithm terminates if maxtime seconds elapsed.
%   minstepsize (1e-10)
%       The algorithm terminates if the linesearch returns a displacement
%       vector (to be retracted) smaller in norm than this value.
%   linesearch (@linesearch or @linesearch_hint)
%       Function handle to a line search function. The options structure is
%       passed to the line search too, so you can pass it parameters. See
%       each line search's documentation for info.
%       If the problem structure includes a line search hint, then the
%       default line search used is @linesearch_hint; otherwise
%       the default is @linesearch.
%       There are other line search algorithms available in
%       /manopt/solvers/linesearch/. For example:
%       - @linesearch_adaptive
%       - @linesearch_constant
%       See their documentation with the help command.
%   statsfun (none)
%       Function handle to a function that will be called after each
%       iteration to provide the opportunity to log additional statistics.
%       They will be returned in the info struct. See the generic Manopt
%       documentation about solvers for further information.
%   stopfun (none)
%       Function handle to a function that will be called at each iteration
%       to provide the opportunity to specify additional stopping criteria.
%       See the generic Manopt documentation about solvers for further
%       information.
%   verbosity (3)
%       Integer number used to tune the amount of output the algorithm
%       generates during execution (mostly as text in the command window).
%       The higher, the more output. 0 means silent.
%   storedepth (2)
%       Maximum number of different points x of the manifold for which a
%       store structure will be kept in memory in the storedb for caching.
%       For the SD algorithm, a store depth of 2 should always be
%       sufficient.
%   hook (none)
%       A function handle which allows the user to change the current point
%       x at the beginning of each iteration, before the stopping criterion
%       is evaluated. See applyHook for help on how to use this option.
%
%
% See also: conjugategradient trustregions manopt/solvers/linesearch manopt/examples

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
%
%   April 3, 2015 (NB):
%       Works with the new StoreDB class system.
%
%   Aug. 2, 2018 (NB):
%       Now using storedb.remove() to keep the cache lean.
%
%   July 19, 2020 (NB):
%       Added support for options.hook.

    
    % Verify that the problem description is sufficient for the solver.
    if ~canGetCost(problem)
        warning('manopt:getCost', ...
                'No cost provided. The algorithm will likely abort.');
    end
    if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
        % Note: we do not give a warning if an approximate gradient is
        % explicitly given in the problem description, as in that case the
        % user seems to be aware of the issue.
        warning('manopt:getGradient:approx', ...
               ['No gradient provided. Using an FD approximation instead (slow).\n' ...
                'It may be necessary to increase options.tolgradnorm.\n' ...
                'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
        problem.approxgrad = approxgradientFD(problem);
    end
    
    % Set local defaults here.
    localdefaults.minstepsize = 1e-10;
    localdefaults.maxiter = 1000;
    localdefaults.tolgradnorm = 1e-6;
    
    % Depending on whether the problem structure specifies a hint for
    % line-search algorithms, choose a default line-search that works on
    % its own (typical) or that uses the hint.
    if ~canGetLinesearch(problem)
        localdefaults.linesearch = @linesearch;
    else
        localdefaults.linesearch = @linesearch_hint;
    end
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    timetic = tic();
    
    % If no initial point x is given by the user, generate one at random.
    if ~exist('x', 'var') || isempty(x)
        x = problem.M.rand();
    end
    
    % Create a store database and get a key for the current x.
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    
    % Compute objective-related quantities for x.
    [cost, grad] = getCostGrad(problem, x, storedb, key);
    gradnorm = problem.M.norm(x, grad);
    
    % Iteration counter.
    % At any point, iter is the number of fully executed iterations so far.
    iter = 0;
    
    % Save stats in a struct array info, and preallocate.
    stats = savestats();
    info(1) = stats;
    info(min(10000, options.maxiter+1)).iter = [];
    
    if options.verbosity >= 2
        fprintf(' iter\t               cost val\t    grad. norm\n');
    end
    
    % Start iterating until stopping criterion triggers.
    while true

        % Display iteration information.
        if options.verbosity >= 2
            fprintf('%5d\t%+.16e\t%.8e\n', iter, cost, gradnorm);
        end
        
        % Start timing this iteration.
        timetic = tic();

        % Apply the hook function if there is one: this allows external code to
        % move x to another point. If the point is changed (indicated by a true
        % value for the boolean 'hooked'), we update our knowledge about x.
        [x, key, info, hooked] = applyHook(problem, x, storedb, key, ...
                                                    options, info, iter+1);
        if hooked
            [cost, grad] = getCostGrad(problem, x, storedb, key);
            gradnorm = problem.M.norm(x, grad);
        end
        
        % Run standard stopping criterion checks.
        [stop, reason] = stoppingcriterion(problem, x, options, ...
                                                             info, iter+1);
        
        % If none triggered, run specific stopping criterion check.
        if ~stop && stats.stepsize < options.minstepsize
            stop = true;
            reason = sprintf(['Last stepsize smaller than minimum '  ...
                              'allowed; options.minstepsize = %g.'], ...
                              options.minstepsize);
        end
    
        if stop
            if options.verbosity >= 1
                fprintf([reason '\n']);
            end
            break;
        end

        % Pick the descent direction as minus the gradient.
        desc_dir = problem.M.lincomb(x, -1, grad);
        
        % Execute the line search.
        [stepsize, newx, newkey, lsstats] = options.linesearch( ...
                             problem, x, desc_dir, cost, -gradnorm^2, ...
                             options, storedb, key);
        
        % Compute the new cost-related quantities for x
        [newcost, newgrad] = getCostGrad(problem, newx, storedb, newkey);
        newgradnorm = problem.M.norm(newx, newgrad);
        
        % Transfer iterate info, remove cache from previous x.
        storedb.removefirstifdifferent(key, newkey);
        x = newx;
        key = newkey;
        cost = newcost;
        grad = newgrad;
        gradnorm = newgradnorm;
        
        % Make sure we don't use too much memory for the store database.
        storedb.purge();
        
        % iter is the number of iterations we have accomplished.
        iter = iter + 1;
        
        % Log statistics for freshly executed iteration.
        stats = savestats();
        info(iter+1) = stats;
        
    end
    
    
    info = info(1:iter+1);

    if options.verbosity >= 1
        fprintf('Total time is %f [s] (excludes statsfun)\n', ...
                info(end).time);
    end
    
    
    
    % Routine in charge of collecting the current iteration stats.
    function stats = savestats()
        stats.iter = iter;
        stats.cost = cost;
        stats.gradnorm = gradnorm;
        if iter == 0
            stats.stepsize = NaN;
            stats.time = toc(timetic);
            stats.linesearch = [];
        else
            stats.stepsize = stepsize;
            stats.time = info(iter).time + toc(timetic);
            stats.linesearch = lsstats;
        end
        stats = applyStatsfun(problem, x, storedb, key, options, stats);
    end
    
end
