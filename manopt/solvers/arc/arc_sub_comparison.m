function [x, cost, info, options] = arc_sub_comparison(problem, x, options)
%Run ARC with Lanczos, but also run CG at each subproblem for comparison.

    M = problem.M;
    
    % Verify that the problem description is sufficient for the solver.
    if ~canGetCost(problem)
        warning('manopt:getCost', ...
                'No cost provided. The algorithm will likely abort.');
    end
    if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
        % Note: we do not give a warning if an approximate gradient is
        % explicitly given in the problem description, as in that case the
        % user seems to be aware of the issue.
        warning('manopt:getGradient:approx', ['No gradient provided. ' ...
                'Using an FD approximation instead (slow).\n' ...
                'It may be necessary to increase options.tolgradnorm.\n'...
                'To disable this warning: ' ...
                'warning(''off'', ''manopt:getGradient:approx'')']);
        problem.approxgrad = approxgradientFD(problem);
    end
    if ~canGetHessian(problem) && ~canGetApproxHessian(problem)
        % Note: we do not give a warning if an approximate Hessian is
        % explicitly given in the problem description, as in that case the
        % user seems to be aware of the issue.
        warning('manopt:getHessian:approx', ['No Hessian provided. ' ...
                'Using an FD approximation instead.\n' ...
                'To disable this warning: ' ...
                'warning(''off'', ''manopt:getHessian:approx'')']);
        problem.approxhess = approxhessianFD(problem);
    end

    % Set local defaults here
    localdefaults.tolgradnorm = 1e-6;
    localdefaults.maxiter = 1000;
    localdefaults.maxtime = inf;
    localdefaults.sigma_min = 1e-7;
    localdefaults.eta_1 = 0.1;
    localdefaults.eta_2 = 0.9;
    localdefaults.gamma_1 = 0.1;
    localdefaults.gamma_2 = 5;
    localdefaults.storedepth = 2;
    localdefaults.subproblemsolver = @arc_lanczos;
    localdefaults.rho_regularization = 1e3;
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    % Default initial sigma_0 is based on the initial Delta_bar of the
    % trustregions solver.
    if ~isfield(options, 'sigma_0')
        if isfield(M, 'typicaldist')
            options.sigma_0 = 100/M.typicaldist();
        else
            options.sigma_0 = 100/sqrt(M.dim());
        end 
    end
    
    
    timetic = tic();
    
    % If no initial point x is given by the user, generate one at random.
    if ~exist('x', 'var') || isempty(x)
        x = M.rand();
    end

    % Create a store database and get a key for the current x.
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();

    % Compute objective-related quantities for x.
    [cost, grad] = getCostGrad(problem, x, storedb, key);
    gradnorm = M.norm(x, grad);
    
    % Initialize regularization parameter.
    sigma = options.sigma_0;

    % Iteration counter.
    % At any point, iter is the number of fully executed iterations so far.
    iter = 0;
    
    % Save stats in a struct array info, and preallocate.
    stats = savestats(problem, x, storedb, key, options);
    info(1) = stats;
    info(min(10000, options.maxiter+1)).iter = [];
    
    if options.verbosity >= 2
        fprintf(' iter\t\t\t\t\tcost val\t\t grad norm       sigma   #Hess\n');
    end
    
    % Iterate until stopping criterion triggers.
    while true

        % Display iteration information.
        if options.verbosity >= 2
            fprintf('%5d\t %+.16e\t %.8e   %.2e', ...
                    iter, cost, gradnorm, sigma);
        end
        
        % Start timing this iteration.
        timetic = tic();
        
        % Run standard stopping criterion checks.
        [stop, reason] = stoppingcriterion(problem, x, options, ...
                                                             info, iter+1);
        
        if stop
            if options.verbosity >= 1
                fprintf(['\n' reason '\n']);
            end
            break;
        end

        % Solve the ARC subproblem.
        % Outputs: eta is the tentative step (it is a tangent vector at x);
        % Heta is the result of applying the Hessian at x along eta (this
        % is often a natural by-product of the subproblem solver);
        % hesscalls is the number of Hessian calls issued in the solver;
        % stop_str is a string describing why the solver terminated; and
        % substats is some statistics about the solver's work to be logged.
        [eta, Heta, hesscalls, stop_str, substats] = ...
             arc_lanczos(problem, x, grad, gradnorm, ...
                                             sigma, options, storedb, key);
        [~, ~, ~, ~, cg_substats] = ...
             arc_conjugate_gradient(problem, x, grad, gradnorm, ...
                                             sigma, options, storedb, key);
        
        etanorm = M.norm(x, eta);

        % Get a tentative next x by retracting the proposed step.
        newx = M.retr(x, eta);
        newkey = storedb.getNewKey();

        % Compute the new cost-related quantities for proposal x.
        % We could just compute the cost here, as the gradient is only
        % necessary if the step is accepted; but we expect most steps are
        % accepted, and sometimes the gradient can be computed faster if it
        % is computed in conjunction with the cost.
        [newcost, newgrad] = getCostGrad(problem, newx, storedb, newkey);

        % Compute a regularized ratio between actual and model improvement.
        rho_num = cost - newcost;
        vec_rho = M.lincomb(x, 1, grad, .5, Heta);
        rho_den = -M.inner(x, eta, vec_rho);
        rho_reg = options.rho_regularization*eps*max(1, abs(cost));
        rho = (rho_num+rho_reg) / (rho_den+rho_reg);
        
        % In principle, the subproblem solver should guarantee rho_den > 0.
        % In practice, it may fail, in which case we reject the step.
        subproblem_failure = (rho_den+rho_reg <= 0);
        if subproblem_failure
            stop_str = sprintf( ...
                 'SUBPROBLEM FAILURE! (Though it returned: %s)', stop_str);
        end
        
        % Determine if the tentative step should be accepted or not.
        if rho >= options.eta_1 && ~subproblem_failure
            accept = true;
            arc_str = 'acc ';
            % We accepted this step: erase cache of the previous point.
            storedb.removefirstifdifferent(key, newkey);
            x = newx;
            key = newkey;
            cost = newcost;
            grad = newgrad;
            gradnorm = M.norm(x, grad);
        else
            accept = false;
            arc_str = 'REJ ';
            % We rejected this step: erase cache of the tentative point.
            storedb.removefirstifdifferent(newkey, key);
        end
        
        % Update the regularization parameter.
        if rho >= options.eta_2 && ~subproblem_failure
            % Very successful step
            arc_str(4) = '-';
            sigma = max(options.sigma_min, options.gamma_1 * sigma);
        elseif rho >= options.eta_1 && ~subproblem_failure
            % Successful step
            arc_str(4) = ' ';
        else
            % Unsuccessful step
            arc_str(4) = '+';
            sigma = options.gamma_2 * sigma;
        end

        % iter is the number of iterations we have completed.
        iter = iter + 1;

        % Make sure we don't use too much memory for the store database.
        storedb.purge();
        
        % Log statistics for freshly executed iteration.
        stats = savestats(problem, x, storedb, key, options);
        info(iter+1) = stats;

        if options.verbosity >= 2
            fprintf('   %5d  %s\n', hesscalls, [arc_str ' ' stop_str]);
        end
        
        % When the subproblem solver fails, it would be nice to have an
        % alternative, such as a slower but more robust solver. For now, we
        % force the solver to terminate when that happens.
        if subproblem_failure
            if options.verbosity >= 1
                fprintf(['\nThe subproblem solver failed to make ' ...
                         'progress even on the model; this is ' ...
                         'likely due to numerical errors.\n']);
            end
            break;
        end
        
    end
    
    % Truncate the struct-array to the part we populated
    info = info(1:iter+1);

    if options.verbosity >= 1
        fprintf('Total time is %f [s] (excludes statsfun)\n', info(end).time);
    end
    
    
    % Routine in charge of collecting the current iteration statistics
    function stats = savestats(problem, x, storedb, key, options)
        
        stats.iter = iter;
        stats.cost = cost;
        stats.gradnorm = gradnorm;
        stats.sigma = sigma;
        
        if iter == 0
            stats.hessiancalls = 0;
            stats.stepsize = NaN;
            stats.time = toc(timetic);
            stats.rho = inf;
            stats.rhonum = NaN;
            stats.rhoden = NaN;
            stats.accepted = true;
            stats.lanczos = struct();
            stats.cg = struct();
        else
            stats.hessiancalls = hesscalls;
            stats.stepsize = etanorm;
            stats.time = info(iter).time + toc(timetic);
            stats.rho = rho;
            stats.rhonum = rho_num;
            stats.rhoden = rho_den;
            stats.accepted = accept;
            stats.lanczos = substats;
            stats.cg = cg_substats;
        end
        
        % Similar to statsfun with trustregions: the x and store passed to
        % statsfun are that of the most recently accepted point after the
        % iteration fully executed.
        stats = applyStatsfun(problem, x, storedb, key, options, stats);
        
    end
    
end
