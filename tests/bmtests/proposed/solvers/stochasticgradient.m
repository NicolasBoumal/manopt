function [x, info, options] = stochasticgradient(problem, x, options)
% Stochastic gradient (SGD) minimization algorithm for Manopt
%
% function [x, info, options] = stochasticgradient(problem)
% function [x, info, options] = stochasticgradient(problem, x)
% function [x, info, options] = stochasticgradient(problem, x, options)
% function [x, info, options] = stochasticgradient(problem, [], options)
%
% Apply the stochasticgradient algorithm to the problem defined
% in the problem structure, starting at x if it is provided (otherwise, at
% a random point on the manifold). To specify options whilst not specifying
% an initial guess, give x as [] (the empty matrix).
%
% The solver mimics other solvers of Manopt with two additonal input
% requirements: problem.ncostterms and problem.partialegrad.
%
% problem.ncostterms has the number of samples, e.g., N samples.
%
% problem.partialegrad takes input a current point of the manifold and
% index of batchsize.
%
% Some of the options of the solver are specifict to this file. Please have
% a look below.


    % Original authors: Bamdev Mishra <bamdevm@gmail.com>,
    %                   Hiroyuki Kasai <kasai@is.uec.ac.jp>, and
    %                   Hiroyuki Sato <hsato@ms.kagu.tus.ac.jp>, 22 April 2016.
    
    % Verify that the problem description is sufficient for the solver.
    if ~canGetPartialGradient(problem)
        warning('manopt:getPartialGradient', ...
            'No partial gradient provided. The algorithm will likely abort.');
    end
    
    % If no initial point x is given by the user, generate one at random.
    if ~exist('x', 'var') || isempty(x)
        x = problem.M.rand();
    end
    
    
    
    % Total number of samples
    N = problem.ncostterms;
    
    % Set local defaults
    localdefaults.maxiter = 100;  % Maximum number of iterations.
    localdefaults.stepsize = 0.1;  % Initial stepsize guess.
    localdefaults.stepsize_type = 'decay'; % Stepsize type. Other possibilities are 'fix' and 'hybrid'.
    localdefaults.stepsize_lambda = 0.1; % lambda is a weighting factor while using stepsize_typ='decay'.
    localdefaults.tolgradnorm = 1.0e-6; % Batch grad norm tolerance.
    localdefaults.batchsize = 1;  % Batchsize.
    localdefaults.verbosity = 0;  % Output verbosity. Other localdefaults are 1 and 2.
    localdefaults.savestatsiter = 1;% Save stats every 1 iteration.
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    
    stepsize0 = options.stepsize;
    batchsize = options.batchsize;
    
    
    % Create a store database and get a key for the current x
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    
    % Compute objective-related quantities for x
    if canGetCost(problem)
        [cost, grad] = getCostGrad(problem, x, storedb, key);
        gradnorm = problem.M.norm(x, grad);
    elseif canGetGradient(problem)
        grad = getGradient(problem, x, storedb, key);
        gradnorm = problem.M.norm(x, grad);
    end
    
    elapsed_time = 0;
    
    % Iteration counter
    iter = 0; % Total number of updates.
    savestatsitercount = 0; % Total number of save stats.
    
    % Save stats in a struct array info, and preallocate.
    stats = savestats();
    info(1) = stats;
    info(min(10000, floor(options.maxiter/options.savestatsiter)+1)).iter = [];
    info(min(10000, floor(options.maxiter/options.savestatsiter)+1)).time = [];
    if canGetCost(problem)
        info(min(10000, floor(options.maxiter/options.savestatsiter)+1)).cost = [];
    end
    if canGetGradient(problem)
        info(min(10000, floor(options.maxiter/options.savestatsiter)+1)).gradnorm = [];
    end
    
    
    
    if options.verbosity > 0 && canGetCost(problem) && canGetGradient(problem)
        fprintf('-------------------------------------------------------\n');
        fprintf('R-SGD:  iter\t               cost val\t    grad. norm\t stepsize\n');
        fprintf('R-SGD:  %5d\t%+.16e\t%.8e\t%.8e\n', 0, cost, gradnorm, stepsize0);
    end
    
    % Draw the samples with replacement.
    perm_idx = randi(N, 1, batchsize*options.maxiter);
    
    
    % Main loop over samples.
    for numupdates = 1 : options.maxiter
        
        % Set start time
        start_time = tic;
        
        % Pick a sample of size batchsize
        start_index = (numupdates-1)*batchsize + 1;
        end_index = batchsize*(min(numupdates, options.maxiter));
        idx_batchsize = perm_idx(start_index : end_index);
        
        
        % Compute gradient on this batch.
        partialgrad = getPartialGradient(problem, x, idx_batchsize, storedb, key);
        
        % Update stepsize
        if strcmp(options.stepsize_type, 'decay')
            stepsize = stepsize0 / (1  + stepsize0 * options.stepsize_lambda * iter); % Decay with O(1/iter).
            
        elseif strcmp(options.stepsize_type, 'fix')
            stepsize = stepsize0; % Fixed stepsize.
            
        elseif strcmp(options.stepsize_type, 'hybrid')
            if numupdates < 5 % Decay stepsize only for the initial few iterations.
                stepsize = stepsize0 / (1  + stepsize0 * options.stepsize_lambda * iter); % Decay with O(1/iter).
            end
            
        else
            error(['Unknown options.stepsize_type. ' ...
                'Should be fix or decay.']);
        end
        
        
        % Update x
        xnew = problem.M.retr(x, partialgrad, -stepsize);
        newkey =  storedb.getNewKey();
        
        % Elapsed time
        elapsed_time = elapsed_time + toc(start_time);
        
        iter = iter + 1; % Total number updates.
        
        
        if mod(iter, options.savestatsiter) == 0 % Save every savestatsiter.
            savestatsitercount = savestatsitercount + 1;
            
            x = xnew;
            key = newkey;
            % Calculate cost, grad, and gradnorm
            if canGetCost(problem)
                [newcost, newgrad] = getCostGrad(problem, xnew, storedb, newkey);
                cost = newcost;
                newgradnorm = problem.M.norm(xnew, newgrad);
                gradnorm = newgradnorm;
            elseif canGetGradient(problem)
                newgrad = getGradient(problem, xnew, storedb, newkey);
                newgradnorm = problem.M.norm(xnew, newgrad);
                gradnorm = newgradnorm;
            end
            
            
            
            % Log statistics for freshly executed iteration
            stats = savestats();
            info(savestatsitercount + 1)= stats;
            
            elapsed_time = 0; % Reset timer
            
            % Print output
            if options.verbosity > 0
                if canGetCost(problem) && canGetGradient(problem)
                    fprintf('R-SGD:  %5d\t%+.16e\t%.8e\t%.8e\n', iter, cost, gradnorm, stepsize);
                elseif canGetCost(problem)
                    fprintf('R-SGD:  %5d\t%+.16e\t          \t%.8e\n', iter, cost, stepsize);
                end
            end
            
            % Stopping criteria
            if gradnorm  <= options.tolgradnorm
                if options.verbosity > 0
                    fprintf('Norm of gradient smaller than %g.\n' ,options.tolgradnorm);
                end
                break;
            end
        end
        
        x = xnew;
        key = newkey;
        
    end
    
    info = info(1: floor(options.maxiter/options.savestatsiter) + 1);
    
    % Save the stats per epoch.
    function stats = savestats()
        stats.iter = iter;
        if canGetCost(problem)
            stats.cost = cost;
        end
        if canGetGradient(problem)
            stats.gradnorm = gradnorm;
        end
        if savestatsitercount == 0
            stats.time = 0;
        else
            stats.time = info(savestatsitercount).time + elapsed_time;
        end
        
        stats = applyStatsfun(problem, x, storedb, key, options, stats);
    end
    
end


