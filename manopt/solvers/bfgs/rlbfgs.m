function [x, cost, info, options] = rlbfgs(problem, x0, options)
% Riemannian BFGS solver for smooth objective function.
%
% function [x, cost, info, options] = rlbfgs(problem)
% function [x, cost, info, options] = rlbfgs(problem, x0)
% function [x, cost, info, options] = rlbfgs(problem, x0, options)
% function [x, cost, info, options] = rlbfgs(problem, [], options)
%
%
% This is Riemannian Limited memory BFGS solver (quasi-Newton method), 
% which aims to minimize the cost function in problem structure problem.cost. 
% It needs gradient of the cost function. There is an filed in options called
% options.memory to specifiy the memory size (number of iteration that the 
% algorithm remembers), and for non-limited memory, simply put options.memory = Inf
%
%
% For a description of the algorithm and theorems offering convergence
% guarantees, see the references below.
%
% The initial iterate is x0 if it is provided. Otherwise, a random point on
% the manifold is picked. To specify options whilst not specifying an
% initial iterate, give x0 as [] (the empty matrix).
%
% The two outputs 'x' and 'cost' are the last reached point on the manifold
% and its cost. 
% 
% The output 'info' is a struct-array which contains information about the
% iterations:
%   iter (integer)
%       The iteration number, or number of steps considered
%       (whether accepted or rejected). The initial guess is 0.
%	cost (double)
%       The corresponding cost value.
%	gradnorm (double)
%       The (Riemannian) norm of the gradient.
%	time (double)
%       The total elapsed time in seconds to reach the corresponding cost.
%	stepsize (double)
%       The size of the step from the previous to the new iterate.
%   accepted (boolean)
%       1 if the current step is accepted in the cautious update. 0 otherwise
%   And possibly additional information logged by options.statsfun.
% For example, type [info.gradnorm] to obtain a vector of the successive
% gradient norms reached at each (outer) iteration.
%
% The options structure is used to overwrite the default values. All
% options have a default value and are hence optional. To force an option
% value, pass an options structure with a field options.optionname, where
% optionname is one of the following and the default value is indicated
% between parentheses:
%
%   tolgradnorm (1e-6)
%       The algorithm terminates if the norm of the gradient drops below
%       this. For well-scaled problems, a rule of thumb is that you can
%       expect to reduce the gradient norm by 8 orders of magnitude
%       (sqrt(eps)) compared to the gradient norm at a "typical" point (a
%       rough initial iterate for example). Further decrease is sometimes
%       possible, but inexact floating point arithmetic will eventually
%       limit the final accuracy. If tolgradnorm is set too low, the
%       algorithm may end up iterating forever (or at least until another
%       stopping criterion triggers).
%   maxiter (1000)
%       The algorithm terminates if maxiter iterations were executed.
%   maxtime (Inf)
%       The algorithm terminates if maxtime seconds elapsed.
%   minstepsize (1e-10)
%     The minimum norm of the tangent vector that points from the current
%     point to the next point. If the norm is less than minstepsize, the 
%     program will terminate.
%   memory (30)
%     The number of previous iterations the program remembers in LBFGS. This is used 
%     to approximate the Hessian at the current point. Because of difficulty
%     of maintaining a representation of hessian in terms of coordinates, and
%     thus a recursive computation for the direction pointing to the next
%     point is done by considering approximating Hessian as an operator that takes
%     a vector and outputs a vector in the tangent space. Theoretically, a
%     vector recurse back memory size number of times and thus memory size 
%     is linear with the time taken to compute directions towards the next
%     point.
%     It can take any value >= 0, or Inf (which will then take value options.maxiter. If
%     options.maxiter has value Inf, then it will take value 10000 with
%     warning displayed).
%   linesearch (@linesearch_hint)
%       Function handle to a line search function. The options structure is
%       passed to the line search too, so you can pass it parameters. See
%       each line search's documentation for info.
%       By default, the intial multiplier tried is alpha = 1. This can be changed
%       with options.linesearch: see the documentation of linesearch_hint.
%   strict_inc_func (@(x) x)
%     The Cautious step needs a real function that has value 0 at x = 0, and 
%     strictly increasing. See details in Wen Huang's paper
%     "A Riemannian BFGS Method without Differentiated Retraction for 
%     Nonconvex Optimization Problems
%   statsfun (none)
%       Function handle to a function that will be called after each
%       iteration to provide the opportunity to log additional statistics.
%       They will be returned in the info struct. See the generic Manopt
%       documentation about solvers for further information. statsfun is
%       called with the point x that was reached last, after the
%       accept/reject decision. See comment below.
%   stopfun (none)
%       Function handle to a function that will be called at each iteration
%       to provide the opportunity to specify additional stopping criteria.
%       See the generic Manopt documentation about solvers for further
%       information.
%   verbosity (2)
%       Integer number used to tune the amount of output the algorithm
%       generates during execution (mostly as text in the command window).
%       The higher, the more output. 0 means silent. 3 and above includes a
%       display of the options structure at the beginning of the execution.
%   debug (false)
%       Set to true to allow the algorithm to perform additional
%       computations for debugging purposes. If a debugging test fails, you
%       will be informed of it, usually via the command window. Be aware
%       that these additional computations appear in the algorithm timings
%       too, and may interfere with operations such as counting the number
%       of cost evaluations, etc. (the debug calls get storedb too).
%   storedepth (30)
%       Maximum number of different points x of the manifold for which a
%       store structure will be kept in memory in the storedb. If the
%       caching features of Manopt are not used, this is irrelevant. If
%       memory usage is an issue, you may try to lower this number.
%       Profiling may then help to investigate if a performance hit was
%       incurred as a result.
% 
%       For a comment about how the info struct-array and statsfun interact 
%       with the notion of accepted / rejected step, see the documention of 
%       trustregions.
%
%
% Please cite the Manopt paper as well as the research paper:
%     @TECHREPORT{HAG2017,
%     author = "Wen Huang and P.-A. Absil and K. A. Gallivan",
%     title = "A Riemannian BFGS Method without Differentiated Retraction for Nonconvex Optimization Problems",
%     institution = "U.C.Louvain",
%     number = "UCL-INMA-2017.04",
%     year = 2017,
%     }
%


% This file is part of Manopt: www.manopt.org.
% Original author: Changshuo Liu, July 19, 2017.
% Contributors: Nicolas Boumal
% Change log: 
%
%   CL, NB July 19, 2017:
%        Finished the first released version


    % Verify that the problem description is sufficient for the solver.
    if ~canGetCost(problem)
        warning('manopt:getCost', ...
            'No cost provided. The algorithm will likely abort.');
    end
    if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
        % Note: we do not give a warning if an approximate gradient is
        % explicitly given in the problem description, as in that case the user
        % seems to be aware of the issue.
        warning('manopt:getGradient:approx', ...
               ['No gradient provided. Using an FD approximation instead (slow).\n' ...
                'This algorithm is not designed to work with inexact gradient: behavior not guaranteed.\n' ...
                'It may be necessary to increase options.tolgradnorm.\n' ...
                'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
        problem.approxgrad = approxgradientFD(problem);
    end
    
    % Local defaults for the program
    localdefaults.minstepsize = 1e-10;
    localdefaults.maxiter = 1000;
    localdefaults.tolgradnorm = 1e-6;
    localdefaults.memory = 30;
    localdefaults.strict_inc_func = @(x) x;
    localdefaults.ls_max_steps  = 25;
    localdefaults.storedepth = 30;
    localdefaults.linesearch = @linesearch_hint;
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    % To make sure memory in range [0, Inf)
    options.memory = max(options.memory, 0);
    if options.memory == Inf
        if isinf(options.maxiter)
            options.memory = 10000;
            warning('rlbfgs:memory',['options.memory and options.maxiter '...
                'are both Inf. This might be too greedy. '...
                'options.memory is now limited to 10000']);
        else
            options.memory = options.maxiter;
        end
    end
    
    M = problem.M;
    
    % Create a random starting point if no starting point
    % is provided.
    if ~exist('x0', 'var')|| isempty(x0)
        xCur = M.rand(); 
    else
        xCur = x0;
    end
    
    timetic = tic();
    
    % Create a store database and get a key for the current x
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    
    % __________Initialization of variables______________
    % Number of iterations since the last restart
    k = 0;  
    % Number of total iteration in BFGS
    iter = 0; 
    % Saves step vectors that points to x_{t+1} from x_{t}
    % for t in range (max(0, iter - min{k, options.memory}), iter].
    % That is, saves up to options.memory number of most 
    % current step vectors. 
    % However, the implementation below does not need stepvectors 
    % in their respective tangent spaces at x_{t}'s, but rather, having 
    % them transported to the most current point's tangent space by vector tranport.
    % For detail of the requirement on the the vector tranport, see the reference. 
    % In implementation, those step vectors are iteratively 
    % transported to most current point's tangent space after every iteration.
    % So at every iteration, it will have this list of vectors in tangent plane
    % of current point.
    sHistory = cell(1, options.memory);
    % Saves the difference between gradient of x_{t+1} and the
    % gradient of x_{t} by transported to x_{t+1}'s tangent space.
    % where t is in range (max(0, iter - min{k, options.memory}), iter].
    % That is, saves up to options.memory number of most 
    % current gradient differences.
    % The implementation process is similar to sHistory.
    yHistory = cell(1, options.memory);
    % rhoHistory{t} is the innerproduct of sHistory{t} and yHistory{t}
    rhoHistory = cell(1, options.memory);
    % Scaling of direction given by getDirection for acceptable step
    alpha = 1; 
    % Scaling of initial matrix, Barzilai-Borwein.
    scaleFactor = 1;
    % Norm of the step
    stepsize = 1;
    % Boolean for whether the step is accepted by Cautious update check
    accepted = 1;
    
    [xCurCost, xCurGradient] = getCostGrad(problem, xCur, storedb, key);
    xCurGradNorm = M.norm(xCur, xCurGradient);
    lsstats = [];
    %A variable to control restarting scheme, see comment below.
    ultimatum = 0;
    
    % Save stats in a struct array info, and preallocate.
    stats = savestats();
    info(1) = stats;
    info(min(10000, options.maxiter+1)).iter = [];
    
    if options.verbosity >= 2
        fprintf(' iter                   cost val            grad. norm           alpha\n');
    end
    
    while (1)
        %------------------------ROUTINE----------------------------

        % Display iteration information
        if options.verbosity >= 2
        %_______Print Information and stop information________
        fprintf('%5d    %+.16e        %.8e      %.4e\n', iter, xCurCost, xCurGradNorm, alpha);
        end
        
        % Start timing this iteration
        timetic = tic();
        
        % Run standard stopping criterion checks
        [stop, reason] = stoppingcriterion(problem, xCur, options, ...
            info, iter+1);
        
        % If none triggered, run specific stopping criterion check
        if ~stop 
            if stats.stepsize < options.minstepsize
                % To avoid infinite loop and to push the search further
                % in case BFGS approximation of Hessian is off towards
                % the end, we erase the memory by setting k = 0;
                % In this way, it starts off like a steepest descent.
                % If even steepest descent does not work, then it is 
                % hopeless and we will terminate.
                if ultimatum == 0
                    if (options.verbosity >= 2)
                        fprintf(['stepsize is too small, restart the bfgs procedure ' ...
                            'with the current point\n']);
                    end
                    k = 0;
                    ultimatum = 1;
                else
                    stop = true;
                    reason = sprintf(['Last stepsize smaller than minimum '  ...
                        'allowed; options.minstepsize = %g.'], ...
                        options.minstepsize);
                end
            else
                ultimatum = 0;
            end
        end  
        
        if stop
            if options.verbosity >= 1
                fprintf([reason '\n']);
            end
            break;
        end

        %--------------------Get Direction-----------------------

        p = getDirection(M, xCur, xCurGradient, sHistory,...
            yHistory, rhoHistory, scaleFactor, min(k, options.memory));

        %--------------------Line Search--------------------------
        [stepsize, xNext, newkey, lsstats] = ...
            linesearch_hint(problem, xCur, p, xCurCost, M.inner(xCur,xCurGradient,p), options, storedb, key);
        
        alpha = stepsize/M.norm(xCur, p);
        step = M.lincomb(xCur, alpha, p);
        
        
        %----------------Updating the next iteration---------------
        [xNextCost, xNextGradient] = getCostGrad(problem, xNext, storedb, newkey);
        sk = M.transp(xCur, xNext, step);
        yk = M.lincomb(xNext, 1, xNextGradient,...
            -1, M.transp(xCur, xNext, xCurGradient));

        inner_sk_yk = M.inner(xNext, yk, sk);
        inner_sk_sk = M.inner(xNext, sk, sk);
        % If cautious step is not accepted, then we do no take the
        % current sk, yk into account. Otherwise, we record it 
        % and use it in approximating hessian.
        % sk, yk are maintained in the most recent point's 
        % tangent space by transport.
        if inner_sk_sk ~= 0 && (inner_sk_yk / inner_sk_sk)>= options.strict_inc_func(xCurGradNorm)
            accepted = 1;
            rhok = 1/inner_sk_yk;
            scaleFactor = inner_sk_yk / M.inner(xNext, yk, yk);
            if (k>= options.memory)
                % sk and yk are saved from 1 to the end
                % with the most currently recorded to the 
                % rightmost hand side of the cells that are
                % occupied. When memory is full, do a shift
                % so that the rightmost is earliest and replace
                % it with the most recent sk, yk.
                for  i = 2:options.memory
                    sHistory{i} = M.transp(xCur, xNext, sHistory{i});
                    yHistory{i} = M.transp(xCur, xNext, yHistory{i});
                end
                if options.memory > 1
                sHistory = sHistory([2:end, 1]);
                yHistory = yHistory([2:end, 1]);
                rhoHistory = rhoHistory([2:end 1]);
                end
                if options.memory > 0
                    sHistory{options.memory} = sk;
                    yHistory{options.memory} = yk;
                    rhoHistory{options.memory} = rhok;
                end
            else
                for  i = 1:k
                    sHistory{i} = M.transp(xCur, xNext, sHistory{i});
                    yHistory{i} = M.transp(xCur, xNext, yHistory{i});
                end
                sHistory{k+1} = sk;
                yHistory{k+1} = yk;
                rhoHistory{k+1} = rhok;
            end
            k = k+1;
        else
            accepted = 0;
            for  i = 1:min(k, options.memory)
                sHistory{i} = M.transp(xCur, xNext, sHistory{i});
                yHistory{i} = M.transp(xCur, xNext, yHistory{i});
            end
        end
        iter = iter + 1;
        xCur = xNext;
        key = newkey;
        xCurGradient = xNextGradient;
        xCurGradNorm = M.norm(xNext, xNextGradient);
        xCurCost = xNextCost;
        
        % Make sure we don't use too much memory for the store database
        storedb.purge();
        
        
        % Log statistics for freshly executed iteration
        stats = savestats();
        info(iter+1) = stats; 
        
    end

    info = info(1:iter+1);
    x = xCur;
    cost = xCurCost;

    if options.verbosity >= 1
        fprintf('Total time is %f [s] (excludes statsfun)\n', ...
                info(end).time);
    end

    % Routine in charge of collecting the current iteration stats
    function stats = savestats()
        stats.iter = iter;
        stats.cost = xCurCost;
        stats.gradnorm = xCurGradNorm;
        if iter == 0
            stats.stepsize = NaN;
            stats.accepted = NaN;
            stats.time = toc(timetic);
        else
            stats.stepsize = stepsize;
            stats.time = info(iter).time + toc(timetic);
            stats.accepted = accepted;
        end
        stats.linesearch = lsstats;
        stats = applyStatsfun(problem, xCur, storedb, key, options, stats);
    end

end

% BFGS step, see Wen's paper for details. This functon basically takes in
% a vector g, and operate inverse approximate Hessian P on it to get 
% Pg, and take negative of it. Due to isometric transport and locking condition
% (see paper), this implementation operates in tangent spaces of the most
% recent point instead of transport this vector iteratively backwards to operate 
% in tangent planes of previous points. Notice that these two conditions are hard
% or expensive to enforce. However, in practice, there is no observed difference
% in them, if your problem requires isotransp, it may be good
% to replace transp with isotransp. There are built in isotransp
% for spherefactory and obliquefactory
function dir = getDirection(M, xCur, xCurGradient, sHistory, yHistory, rhoHistory, scaleFactor, k)
    q = xCurGradient;
    inner_s_q = zeros(1, k);
    for i = k : -1 : 1
        inner_s_q(1, i) = rhoHistory{i} * M.inner(xCur, sHistory{i}, q);
        q = M.lincomb(xCur, 1, q, -inner_s_q(1, i), yHistory{i});
    end
    r = M.lincomb(xCur, scaleFactor, q);
    for i = 1 : k
         omega = rhoHistory{i} * M.inner(xCur, yHistory{i}, r);
         r = M.lincomb(xCur, 1, r, inner_s_q(1, i)-omega, sHistory{i});
    end
    dir = M.lincomb(xCur, -1, r);
end
