function [xbest fbest info] = pso(problem, x, options)
% Particle swarm optimization (PSO) for derivative-free minimization.
%
% function [x cost info] = pso(problem)
% function [x cost info] = pso(problem, x0)
% function [x cost info] = pso(problem, x0, options)
%
% Apply the Particle Swarm Optimization minimization algorithm to 
% the problem defined in the problem structure, starting with the
% population x0 if it is provided (otherwise, a random population on the
% manifold is generated). A population is a cell containing points on the
% manifold. The number of elements in the cell must match the parameter
% options.populationsize.
%
% To specify options whilst not specifying an initial guess, give x0 as []
% (the empty matrix).
%
% None of the options are mandatory. See the documentation for details.
%
% Based on the original PSO description in
%   http://particleswarm.info/nn951942.ps.

% This file is part of Manopt: www.manopt.org.
% Original author: Pierre Borckmans, Dec. 30, 2012.
% Contributors: 
% Change log: 


    % Import necessary tools etc. here
    import manopt.privatetools.*;
    
    % Verify that the problem description is sufficient for the solver.
    if ~canGetCost(problem)
        warning('manopt:getCost', ...
                'No cost provided. The algorithm will likely abort.');  
    end
    
    % Dimension of the manifold
    dim = problem.M.dim();

    % Set local defaults here
    % localdefaults.epsilon = 1e-6;
    localdefaults.storedepth = 0;                   % no need for caching
    localdefaults.maxcostevals = max(5000, 2*dim);
    localdefaults.maxiter = max(500, 4*dim);
    
    localdefaults.populationsize = min(40,10*dim);
    localdefaults.nostalgia = 1.4;
    localdefaults.social = 1.4;
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    % Create a store database
    storedb = struct();
    
    % Start timing for initialization
    timetic = tic();
    
    % If no initial population x is given by the user, generate one at random.
    if ~exist('x', 'var') || isempty(x)
        x = cell(options.populationsize, 1);
        for i = 1 : options.populationsize
            x{i} = problem.M.rand();
        end
    end
    % Initialize personal best positions to the initial population
    y = x;
    % Save a copy of the swarm at the previous iteration
    xprev = x;
    
    % Initialize velocities for each particle
    v = cell(options.populationsize, 1);
    for i = 1 : options.populationsize
        % random velocity to improve initial exploration
        v{i} = problem.M.randvec(x{i});
        % or null velocity
        % v{i} = problem.M.zerovec();
    end
        
    % Compute cost for each particle xi, 
    % initialize personal best costs,
    % and setup a function evaluations counter.
    costs = zeros(options.populationsize, 1);
    for i = 1 : options.populationsize
        [costs(i) storedb] = getCost(problem, x{i}, storedb);
        
    end
    fy = costs;
    costevals = options.populationsize;
    
    % Search the best particle and store its cost/position
    [fbest imin] = min(costs);
    xbest = x{imin};
    
    % Iteration counter (at any point, iter is the number of fully executed
    % iterations so far)
    iter = 0;
    
    % Save stats in a struct array info, and preallocate
    % (see http://people.csail.mit.edu/jskelly/blog/?x=entry:entry091030-033941)
    % savestats will be called twice for the initial iterate (number 0),
    % which is unfortunate, but not problematic.
    stats = savestats();
    info(1) = stats;
    info(min(10000, options.maxiter+1)).iter = [];
    
    % Start iterating until stopping criterion triggers
    while true
        
        stats = savestats();
        info(iter+1) = stats; %#ok<AGROW>
        iter = iter + 1;
        
        % Make sure we don't use too much memory for the store database
        storedb = purgeStoredb(storedb, options.storedepth);
        
        % Log / display iteration information here.
        if options.verbosity >= 2
            fprintf('Cost evals: %7d\tBest cost: %+.8e\n', costevals, fbest);
        end
        
        % Start timing this iteration
        timetic = tic();
        
        % Run standard stopping criterion checks
        [stop reason] = stoppingcriterion(problem, x, options, info, iter);
    
        if stop
            if options.verbosity >= 1
                fprintf([reason '\n']);
            end
            break;
        end
        
        
        % Compute the inertia factor 
        % (linearly decreasing from .9 to .4, from iter=0 to maxiter)
        w = 0.4 + 0.5*(1-iter/options.maxiter);

        % Compute velocities
        for i = 1 : options.populationsize
            % Get the position and past best position of particle i
            xi = x{i};
            yi = y{i}; 
            % Get the previous position and velocity of particle i
            xiprev = xprev{i};
            vi = v{i};

            % Compute new velocity of particle i, 
            % composed of 3 contributions
            inertia = w * problem.M.transp(xiprev, xi, vi);
            nostalgia = rand(1) * options.nostalgia * problem.M.log(xi, yi);
            social = rand(1) * options.social * problem.M.log(xi, xbest);       
            v{i} = inertia + nostalgia + social;
        end
        
        % Backup the current swarm positions
        xprev = x;
        
        % Update positions, personal bests and global best
        for i = 1 : options.populationsize
            % compute new position of particle i
            x{i} = problem.M.retr(x{i}, v{i});
            % compute new cost of particle i
            [fxi storedb] = getCost(problem, x{i}, storedb);
            costevals = costevals + 1;
    
            % update costs of the swarm
            costs(i) = fxi;
            % update self-best if necessary
            if fxi < fy(i)
                % update self-best cost and position
                fy(i) = fxi;
                y{i} = x{i};
                % update global-best if necessary
                if fy(i) < fbest
                    fbest = fy(i);
                    xbest = y{i};
                end
            end
        end
    end
    
    
    info = info(1:iter);
    
    
    
    
    % Routine in charge of collecting the current iteration stats
    function stats = savestats()
        stats.iter = iter;
        stats.cost = fbest;
        stats.costevals = costevals;
        stats.x = x;
        stats.v = v;
        stats.xbest = xbest;
        if iter == 0
            stats.time = toc(timetic);
        else
            stats.time = info(iter).time + toc(timetic);
        end
        stats = manopt.privatetools.applyStatsfun(...
                                          problem, x, storedb, options, stats);
    end
    
end
