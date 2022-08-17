function [eta, Heta, print_str, stats] = tCG_rejectedstep(problem, subprobleminput, options, storedb, key)
% Function that mimics trs_tCG_cached's behaviour with existing cache.
% 
% function [eta, Heta, print_str, stats] = tCG_rejectedstep(problem, subprobleminput, options, storedb, key)
%
% Upon step rejection in trustregions.m, instead of running the entire tCG
% loop again, the stored information from a previous call to trs_tCG_cached
% is sufficient to compute the next proposed step. tCG_rejectedstep
% processes this stored information and computes the next step.
%
% There can be two situations:
% 
% 1. The same eta and Heta as the previous tCG loop is returned and 
% trustregions.m decreases Delta. (either d_Hd <= 0 or store_last is 
% used)
% 
% 2. A new eta and Heta is returned when some previously calculated eta_new 
% from store_iters satisfies normsq := <eta_new,eta_new>_x >= Delta^2 
% at the current Delta (exceeding trust region). Then the returned point is
% the Steihaug–Toint point calculated using the eta computed before
% eta_new.
%
% Refer to trs_tCG_cached for a description of the inputs and outputs.
%
% See also: trustregions trs_tCG_cached trs_tCG

% This file is part of Manopt: www.manopt.org.
% Original author: Victor Liao, Jun. 24, 2022.

    x = subprobleminput.x;
    Delta = subprobleminput.Delta;

    lincomb = @(a, u, b, v) problem.M.lincomb(x, a, u, b, v);

    store = storedb.get(key);
    store_iters = store.store_iters;
    stats.memorytCG_MB = getsize(store_iters(1))/1024^2 * length(store_iters);
    numstored = length(store_iters);
    if isfield(store, 'store_last')
        store_last = store.store_last;
        stats.memorytCG_MB = stats.memorytCG_MB + getsize(store_last)/1024^2;
        numstored = numstored + 1;
    end
    % get amount of memory that is currently cached.

    for i = 1:length(store_iters)
        normsq = store_iters(i).normsq;
        d_Hd = store_iters(i).d_Hd;
        if d_Hd <= 0 || normsq >= Delta^2
            % We exit after computing new eta, Heta dependent on Delta
            e_Pe = store_iters(i).e_Pe;
            e_Pd = store_iters(i).e_Pd;
            d_Pd = store_iters(i).d_Pd;
            eta = store_iters(i).eta;
            mdelta = store_iters(i).mdelta;
            Hmdelta = store_iters(i).Hmdelta;
            Heta = store_iters(i).Heta;
            
            tau = (-e_Pd + sqrt(e_Pd*e_Pd + d_Pd*(Delta^2-e_Pe))) / d_Pd;
            if options.debug > 2
                fprintf('DBG:     tau  : %e\n', tau);
            end
            eta  = lincomb(1,  eta, -tau,  mdelta);
            
            % If only a nonlinear Hessian approximation is available, this is
            % only approximately correct, but saves an additional Hessian call.
            Heta = lincomb(1, Heta, -tau, Hmdelta);
            
            % Technically, we may want to verify that this new eta is indeed
            % better than the previous eta before returning it (this is always
            % the case if the Hessian approximation is linear, but I am unsure
            % whether it is the case or not for nonlinear approximations.)
            % At any rate, the impact should be limited, so in the interest of
            % code conciseness (if we can still hope for that), we omit this.
            
            if d_Hd <= 0
                stopreason_str = 'negative curvature';
            else
                stopreason_str = 'exceeded trust region';
            end
            
            stats.limitedbyTR = true;
            stats.numinner = store_iters(i).numinner;
            stats.hessvecevals = 0;

            if options.verbosity == 2
                print_str = sprintf('%9d   %9d   %9d   %s', stats.numinner, 0, numstored, stopreason_str);
            elseif options.verbosity > 2
                print_str = sprintf('%9d   %9d   %9d   %9.2f   %s', stats.numinner, 0, numstored, stats.memorytCG_MB, stopreason_str);
            end

            return;
        end
    end

    % If no stored struct in store_iters satisfies negative curvature or 
    % violates the trust-region radius we exit with last eta, Heta and
    % limitedbyTR = false from store_last.
    eta = store_last.eta;
    Heta = store_last.Heta;
    stats.limitedbyTR = false;
    stats.numinner = store_last.numinner;
    stats.hessvecevals = 0;
    if options.verbosity == 2
        print_str = sprintf('%9d   %9d   %9d   %s', stats.numinner, 0, numstored, store_last.stopreason_str);
    elseif options.verbosity > 2
        print_str = sprintf('%9d   %9d   %9d   %9.2f   %s', stats.numinner, 0, numstored, stats.memorytCG_MB, store_last.stopreason_str);
    end
