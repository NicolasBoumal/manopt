function [x, info, options] = stochasticgradient(problem, x, options)
% Stochastic gradient (SG) minimization algorithm for Manopt.
%
% function [x, info, options] = stochasticgradient(problem)
% function [x, info, options] = stochasticgradient(problem, x)
% function [x, info, options] = stochasticgradient(problem, x, options)
% function [x, info, options] = stochasticgradient(problem, [], options)
%
% Apply the Riemannian stochastic gradient algorithm to the problem defined
% in the problem structure, starting at x if it is provided (otherwise, at
% a random point on the manifold). To specify options whilst not specifying
% an initial guess, give x as [] (the empty matrix).
%
% The problem structure must contain the following fields:
%
%  problem.M:
%       Defines the manifold to optimize over, given by a factory.
%
%  problem.partialgrad or problem.partialegrad (or equivalent)
%       Describes the partial gradients of the cost function. If the cost
%       function is of the form f(x) = sum_{k=1}^N f_k(x),
%       then partialegrad(x, K) = sum_{k \in K} grad f_k(x).
%       As usual, partialgrad must define the Riemannian gradient, whereas
%       partialegrad defines a Euclidean (classical) gradient which will be
%       converted automatically to a Riemannian gradient. Use the tool
%       checkgradient(problem) to check it.
%
%  problem.ncostterms
%       An integer specifying how many terms are in the cost function (in
%       the example above, that would be N.)
%
% Importantly, the cost function itself needs not be specified.
%
% Some of the options of the solver are specifict to this file. Please have
% a look inside the code.
%
% See also: steepestdescent stochasticvariancereducedgradient

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra <bamdevm@gmail.com>,
%                  Hiroyuki Kasai <kasai@is.uec.ac.jp>, and
%                  Hiroyuki Sato <hsato@ms.kagu.tus.ac.jp>, 22 April 2016.
% Contributors: Nicolas Boumal
% Change log: 
    
    % Verify that the problem description is sufficient for the solver.
    if ~canGetPartialGradient(problem)
        warning('manopt:getPartialGradient', ...
            'No partial gradient provided. The algorithm will likely abort.');
    end
    
    % If no initial point x is given by the user, generate one at random.
    if ~exist('x', 'var') || isempty(x)
        x = problem.M.rand();
    end
    
   
    % Set local defaults
    localdefaults.maxiter = 1000;     % Maximum number of iterations.
    localdefaults.tolgradnorm = 1e-6; % Batch gradient norm tolerance.
    localdefaults.batchsize = 1;      % Batchsize.
    localdefaults.verbosity = 0;      % Output verbosity (higher -> more output)
    localdefaults.savestatsiter = 1;  % Save stats every # iteration.
    
    %%% QUESTION OF NICOLAS: Can we make this a lot more flexible and just
    %%% have stepsize be a function handle that takes as input the
    %%% iteration and batch number, and perhaps the previous value?
    %%% We can then have a default function handle here. But then it is a
    %%% bit more complicated to keep the same type of step choice while
    %%% only changing the rate. Let's talk about this. Could simplify code.
    localdefaults.stepsize_init = 0.1;  % Initial stepsize guess.
    localdefaults.stepsize_type = 'decay'; % Stepsize type. Other possibilities are 'fix' and 'hybrid'.
    localdefaults.stepsize_lambda = 0.1; % lambda is a weighting factor while using stepsize_typ='decay'.
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    
    stepsize_init = options.stepsize_init;
    batchsize = options.batchsize;
    
    
    % Create a store database and get a key for the current x
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    
    % Compute objective-related quantities for x
    %%% NICOLAS: I do not think this should be the default behavior.
    if canGetCost(problem)
        [cost, grad] = getCostGrad(problem, x, storedb, key);
        gradnorm = problem.M.norm(x, grad);
    elseif canGetGradient(problem)
        % NICOLAS: this will always be true, because partialgrad is enough
        % to compute the gradient. Also think this shouldn't be default
        % behavior.
        grad = getGradient(problem, x, storedb, key);
        gradnorm = problem.M.norm(x, grad);
    end
    
    elapsed_time = 0;
    
    % Iteration counter
    iter = 0; % Total number of updates.
    savestatsitercount = 0; % Total number of saved stats at this point.
    %%% The semantics for savestatsitercount was wrong, because we compute
    %%% stats just below and we do not increment it. I think it should be
    %%% incremented AND there are modifications below to be implemented
    %%% too.
    
    % Save stats in a struct array info, and preallocate.
    %%% NICOLAS: I simplified here to always allocate 100000 and never less;
    %%% after all, if we're doing SGD, there is a big dataset somewhere in
    %%% RAM already, so this is inconsequential (we may need more,
    %%% actually.) This here will be a few Mb..
    stats = savestats();
    info(1) = stats;
    savestatsitercount = savestatsitercount+1; %%% NICO added
    
    info(100000).iter = [];
    info(100000).time = [];
    if canGetCost(problem)
        info(100000).cost = []; %%% NICOLAS to be checked if still needed
    end
    if canGetGradient(problem) %%% NICOLAS also to be checked if still needed
        info(100000).gradnorm = [];
    end
    
    
    %%% NICOLAS: in my view, by default, the solver should output
    %%% statistics that are inherent to its own functioning, so that it
    %%% doesn't add significantly to the computational load. Let's talk
    %%% about what should be displayed.
    if options.verbosity > 0 && canGetCost(problem) && canGetGradient(problem)
        fprintf('-------------------------------------------------------\n');
        fprintf('iter\t               cost val\t    grad. norm\t stepsize\n');
        fprintf('%5d\t%+.16e\t%.8e\t%.8e\n', 0, cost, gradnorm, stepsize_init);
    end
    
    
    % Main loop over samples.
    for numupdates = 1 : options.maxiter
        
        % Set start time
        start_time = tic;
        
        % Draw the samples with replacement.
        idx_batch = randi(problem.ncostterms, batchsize, 1);
        
        % Compute gradient on this batch.
        partialgrad = getPartialGradient(problem, x, idx_batch, storedb, key);
        
        % Update stepsize
        switch lower(options.stepsize_type)
            case 'decay' % Decay as O(1/iter).
                stepsize = stepsize_init / (1  + stepsize_init * options.stepsize_lambda * iter);
            
            case {'fix', 'fixed'} % Fixed stepsize.
                stepsize = stepsize_init;
                
            case 'hybrid'
                if numupdates < 5 % Decay stepsize only for the initial few iterations.
                    stepsize = stepsize_init / (1  + stepsize_init * options.stepsize_lambda * iter);
                end
                
            otherwise
                error('Unknown options.stepsize_type. Should be ''fix'' or ''decay''.');
        end
        
        
        % Compute the new point and give it a key
        xnew = problem.M.retr(x, partialgrad, -stepsize);
        newkey = storedb.getNewKey();
        
        % Make the step
        x = xnew;
        key = newkey;
        
        % Total number of completed steps
        iter = iter + 1;
        
        % Elapsed time
        elapsed_time = elapsed_time + toc(start_time);
        
        
        % Save statistics only every savestatsiter iterations.
        if mod(iter, options.savestatsiter) == 0
            
            %%%NICOLAS same discussion
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
            info(savestatsitercount+1) = stats;
            savestatsitercount = savestatsitercount+1;
            
            elapsed_time = 0; % Reset timer
            
            % Print output
            if options.verbosity > 0
                if canGetCost(problem) && canGetGradient(problem)
                    fprintf('%5d\t%+.16e\t%.8e\t  %.8e\n', iter, cost, gradnorm, stepsize);
                end
            end

            % Run standard stopping criterion checks
            [stop, reason] = stoppingcriterion(problem, x, ...
                                        options, info, savestatsitercount);
            if stop
                if options.verbosity >= 1
                    fprintf([reason '\n']);
                end
                break;
            end
        end
        
    end
    
    info = info(1:savestatsitercount);
    
    
    % Helper function to collect statistics to be saved at index
    % savestatsitercount+1 in info.
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


