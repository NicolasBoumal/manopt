function [x cost info] = conjugategradient(problem, x, options)
% Conjugate gradient minimization algorithm for Manopt.
%
% function [x cost info] = conjugategradient(problem)
% function [x cost info] = conjugategradient(problem, x0)
% function [x cost info] = conjugategradient(problem, x0, options)
%
% Apply the conjugate gradient minimization algorithm to the problem
% defined in the problem structure, starting at x0 if it is provided
% (otherwise, at a random point on the manifold). To specify options whilst
% not specifying an initial guess, give x0 as [] (the empty matrix).
%
% None of the options are mandatory. See the documentation for details.
%
% For input/output descriptions, stopping criteria, help on picking a line
% search algorithm etc, see the help for steepestdescent.
%
% See also: steepestdescent linesearch
%

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, Dec. 30, 2012.
% Contributors: 
% Change log: 


% Import necessary tools etc. here
import manopt.privatetools.*;

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
localdefaults.linesearch = @manopt.solvers.linesearch.linesearch_adaptive;
localdefaults.beta_type = 'P-R'; % by BM
localdefaults.orth_value = Inf; % by BM as suggested in Nocedal and Wright

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

% Save stats in a struct array info and preallocate,
% see http://people.csail.mit.edu/jskelly/blog/?x=entry:entry091030-033941
stats = savestats();
info(1) = stats;
info(min(10000, options.maxiter+1)).iter = [];

% Initial linesearch memory
lsmem = [];


if options.verbosity >= 2
    fprintf(' iter\t    cost val\t grad. norm\n');
end

% Compute a first descent direction (not normalized)
desc_dir = problem.M.lincomb(x, -1, grad);


% Start iterating until stopping criterion triggers
while true
    
    % Display iteration information
    if options.verbosity >= 2
        fprintf('%5d\t%+.4e\t%.4e\n', iter, cost, gradnorm);
    end
    
    % Start timing this iteration
    timetic = tic();
    
    % Run standard stopping criterion checks
    [stop reason] = stoppingcriterion(problem, x, options, info, iter+1);
    
    % Run specific stopping criterion check
    if ~stop && abs(stats.stepsize) < options.minstepsize
        stop = true;
        reason = 'Last stepsize smaller than minimum allowed.';
    end
    
    if stop
        if options.verbosity >= 1
            fprintf([reason '\n']);
        end
        break;
    end
    
    
    % The line search algorithms require the directional derivative of the
    % cost at the current point x along the search direction.
    df0 = problem.M.inner(x, grad, desc_dir);
        
    % If we didn't get a descent direction, reverse it or switch to the
    % negative gradient. Equivalent to resetting the CG direction, i.e.,
    % discarding the past information.
    if df0 > 0
        
%         % Either we reverse it.
%         if options.verbosity >= 3
%             fprintf(['Conjugate gradient warning: got an ascent direction ' ...
%                      '(df0 = %2e), went the other way.\n'], df0);
%         end
%         df0 = -df0;
%         desc_dir = problem.M.lincomb(x, -1, desc_dir);

        % Or we switch to the negative gradient direction.
        if options.verbosity >= 3
            fprintf(['Conjugate gradient warning: got an ascent direction ' ...
                     '(df0 = %2e), reset to the steepest descent direction.\n'], df0);
        end
        desc_dir = problem.M.lincomb(x, -1, grad); % Reset to negative gradient. Erase memory.
        df0 = -(gradnorm^2);
        
    end
    
    % The line search algorithms require a normalized search direction and
    % directional derivative
    norm_dir = problem.M.norm(x, desc_dir);
    desc_dir_normalized = problem.M.lincomb(x, 1/norm_dir, desc_dir);
    df0 = df0/norm_dir;
  
    % Execute line search
    [stepsize newx storedb lsmem lsstats] = options.linesearch(...
      problem, x, desc_dir_normalized, cost, df0, options, storedb, lsmem);

    
    % Compute the new objective-related quantities for x
    [newcost newgrad storedb] = getCostGrad(problem, newx, storedb);
    newgradnorm = problem.M.norm(newx, newgrad);
    
    
    % CG scheme here by BM
    if strcmpi(options.beta_type, 'steep'), % Gradient Descent
        desc_dir = problem.M.lincomb(x, -1, newgrad);
    else
        grad_old = problem.M.transp(x,newx, grad);
        orth_grads = problem.M.inner(newx, grad_old, newgrad)/(newgradnorm^2);
        
        if abs(orth_grads) >= options.orth_value,
            desc_dir = problem.M.lincomb(x, -1, newgrad);
        else % Compute the CG modification
            desc_dir = problem.M.transp(x, newx, desc_dir);
            if strcmp(options.beta_type, 'F-R')  % Fletcher-Reeves
                beta = (newgradnorm / gradnorm)^2;
                
            elseif strcmp(options.beta_type, 'P-R')  % Polak-Ribiere+
                % vector grad(new) - transported grad(current)
                diff = problem.M.lincomb(newx, 1, newgrad, -1, grad_old);
                ip_diff = problem.M.inner(newx,newgrad,diff);
                beta = ip_diff/(gradnorm^2);
                beta = max(0, beta);
                
           elseif strcmp(options.beta_type, 'D')  % Daniel from AMS08
                % Requires Second-order information
                Hessian_desc_dir = getHessian(problem, newx, desc_dir, storedb);
                numo = problem.M.inner(newx, Hessian_desc_dir, newgrad);
                deno = problem.M.inner(newx, Hessian_desc_dir, desc_dir);
                beta = numo/deno;
                
                % Avoid negative values of beta all together. 
                % Reference? Seems to perform better. by BM
                beta = max(0, beta);
                
            elseif strcmp(options.beta_type, 'H-S')  % Hestenes-Stiefel+
                diff = problem.M.lincomb(newx, 1, newgrad, -1, grad_old);
                ip_diff = problem.M.inner(newx,newgrad,diff);
                beta = ip_diff / problem.M.inner(newx, diff, desc_dir);
                beta = max(0, beta);

            elseif strcmp(options.beta_type, 'H-Z') % Hager-Zhang+
                diff = problem.M.lincomb(newx, 1, newgrad, -1, grad_old);
                deno = problem.M.inner(newx, diff, desc_dir);
                ip_diff_diff = problem.M.inner(newx,diff,diff);
                desc_dir_mod = problem.M.lincomb(newx, 1, diff, -2*ip_diff_diff/deno, desc_dir);
                numo = problem.M.inner(newx, newgrad, desc_dir_mod);
                beta = numo/deno;
                
                % Robustness
                norm_desc_dir = problem.M.norm(newx, desc_dir); 
                eta_HZ = -1/(norm_desc_dir * min(0.01, gradnorm));
                beta = max(beta,  eta_HZ);

            else
                error(['Unknown options.beta_type. ' ...
                       'Should be steep, F-R, P-R, D, H-S or H-Z. ']);
            end
            desc_dir = problem.M.lincomb(newx, -1, newgrad, beta, desc_dir);
        end
        
    end
    
    % Make sure we don't use too much memory for the store database.
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
    fprintf('Total time is %f [s] (excludes statsfun)\n', info(end).time);
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
        if isfield(options, 'statsfun')
            stats = options.statsfun(problem, x, stats);
        end
    end

end


