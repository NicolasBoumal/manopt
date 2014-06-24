function [x cost info] = conjugategradient(problem, x, options)
% Conjugate gradient minimization algorithm for Manopt.
%
% function [x cost info] = conjugategradient(problem)
% function [x cost info] = conjugategradient(problem, x0)
% function [x cost info] = conjugategradient(problem, x0, options)
% function [x cost info] = conjugategradient(problem, [], options)
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
% Contributors: Nicolas Boumal
% Change log: 
%  - Added preconditioner support : see Section 8 in
%    https://www.math.lsu.edu/~hozhang/papers/cgsurvey.pdf
%    NB, March 14, 2013.


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
localdefaults.linesearch = @linesearch_adaptive;
% Changed by NB : H-S has the "auto restart" property.
% See Hager-Zhang 2005 survey about CG methods.
localdefaults.beta_type = 'H-S';
localdefaults.orth_value = Inf; % by BM as suggested in Nocedal and Wright

% Merge global and local defaults, then merge w/ user options, if any.
localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
if ~exist('options', 'var') || isempty(options)
    options = struct();
end
options = mergeOptions(localdefaults, options);

% for convenience
inner = problem.M.inner;

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
[Pgrad storedb] = getPrecon(problem, x, grad, storedb);
gradPgrad = inner(x, grad, Pgrad);

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
desc_dir = problem.M.lincomb(x, -1, Pgrad);


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
    df0 = inner(x, grad, desc_dir);
        
    % If we didn't get a descent direction, reverse it or switch to the
    % negative gradient. Equivalent to resetting the CG direction, i.e.,
    % discarding the past information.
    if df0 >= 0
        
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
                     '(df0 = %2e), reset to the (preconditioned) steepest ' ...
                     'descent direction.\n'], df0);
        end
        desc_dir = problem.M.lincomb(x, -1, Pgrad); % Reset to negative gradient. Erase memory.
        df0 = -gradPgrad;
        
    end
    
    % The line search algorithms require a normalized search direction and
    % directional derivative
    desc_dir_norm = problem.M.norm(x, desc_dir);
    normd_desc_dir = problem.M.lincomb(x, 1/desc_dir_norm, desc_dir);
    normd_df0 = df0/desc_dir_norm;
  
    % Execute line search
    [stepsize newx storedb lsmem lsstats] = options.linesearch(...
     problem, x, normd_desc_dir, cost, normd_df0, options, storedb, lsmem);

    
    % Compute the new objective-related quantities for x
    [newcost newgrad storedb] = getCostGrad(problem, newx, storedb);
    newgradnorm = problem.M.norm(newx, newgrad);
    [Pnewgrad storedb] = getPrecon(problem, x, newgrad, storedb);
    newgradPnewgrad = inner(newx, newgrad, Pnewgrad);
    
    
    % CG scheme here by BM
    %
    % NB TODO : https://www.math.lsu.edu/~hozhang/papers/cgsurvey.pdf
    % List all these beta schemes ; choose the default one wisely, in
    % particular with respect to jamming: ./gradnorm^2 might not be the
    % better choice for a default.
    if strcmpi(options.beta_type, 'steep'), % Gradient Descent
        
        desc_dir = problem.M.lincomb(x, -1, Pnewgrad);
        
    else
        
        oldgrad = problem.M.transp(x, newx, grad);
%         orth_grads = inner(newx, oldgrad, newgrad)/(newgradnorm^2);
        orth_grads = inner(newx, oldgrad, Pnewgrad)/newgradPnewgrad;
        
        % Powell's restart strategy (see page 12 of Hager and Zhang's
        % survey on conjugate gradient methods, for example)
        if abs(orth_grads) >= options.orth_value,
            desc_dir = problem.M.lincomb(x, -1, Pnewgrad);
            
        else % Compute the CG modification
            
            desc_dir = problem.M.transp(x, newx, desc_dir);
            
            if strcmp(options.beta_type, 'F-R')  % Fletcher-Reeves
                beta = newgradPnewgrad / gradPgrad;
                
            elseif strcmp(options.beta_type, 'P-R')  % Polak-Ribiere+
                % vector grad(new) - transported grad(current)
                diff = problem.M.lincomb(newx, 1, newgrad, -1, oldgrad);
                ip_diff = inner(newx, Pnewgrad, diff);
                beta = ip_diff/gradPgrad;
                beta = max(0, beta);
                
           elseif strcmp(options.beta_type, 'D')  % Daniel from AMS08
                % Requires Second-order information
                % Not adapted for precon : would require the
                % change-of-varialbe matrix and not just the precon. Use
                % trustregions instead, with precon.
                [Hessian_desc_dir storedb] = ...
                              getHessian(problem, newx, desc_dir, storedb);
                numo = inner(newx, Hessian_desc_dir, newgrad);
                deno = inner(newx, Hessian_desc_dir, desc_dir);
                beta = numo/deno;
                
                % Avoid negative values of beta all together. 
                % Reference? Seems to perform better. by BM
                beta = max(0, beta);
                
            elseif strcmp(options.beta_type, 'H-S')  % Hestenes-Stiefel+
                diff = problem.M.lincomb(newx, 1, newgrad, -1, oldgrad);
                ip_diff = inner(newx, Pnewgrad, diff);
                beta = ip_diff / inner(newx, diff, desc_dir);
                beta = max(0, beta);

            elseif strcmp(options.beta_type, 'H-Z') % Hager-Zhang+
                diff = problem.M.lincomb(newx, 1, newgrad, -1, oldgrad);
                Poldgrad = problem.M.transp(x, newx, Pgrad);
                Pdiff = Pnewgrad - Poldgrad;
                deno = inner(newx, diff, desc_dir);
%                 ip_diff_diff = inner(newx,diff,diff);
%                 desc_dir_mod = problem.M.lincomb(newx, 1, diff, -2*ip_diff_diff/deno, desc_dir);
%                 numo = inner(newx, newgrad, desc_dir_mod);
                numo = inner(newx, diff, Pnewgrad);
                numo = numo - 2*inner(newx, diff, Pdiff)*inner(newx, desc_dir, newgrad)/deno;
                beta = numo/deno;
                
                % Robustness [NB : what is this? ref?]
                % TODO : do we need this? what is this 0.01? this assumes
                % some scaling of the problem, no?
%                 desc_dir_norm = problem.M.norm(newx, desc_dir); % already
%                 computed?
                eta_HZ = -1/(desc_dir_norm * min(0.01, gradnorm));
                beta = max(beta,  eta_HZ);

            else
                error(['Unknown options.beta_type. ' ...
                       'Should be steep, F-R, P-R, D, H-S or H-Z. ']);
            end
            desc_dir = problem.M.lincomb(newx, -1, Pnewgrad, beta, desc_dir);
        end
        
    end
    
    % Make sure we don't use too much memory for the store database.
    storedb = purgeStoredb(storedb, options.storedepth);
    
    % Update iterate info
    x = newx;
    cost = newcost;
    grad = newgrad;
    Pgrad = Pnewgrad;
    gradnorm = newgradnorm;
    gradPgrad = newgradPnewgrad;
    
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
        stats = applyStatsfun(problem, x, storedb, options, stats);
    end

end


