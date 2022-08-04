function [eta, Heta, print_str, savedoutput] = tCG_rejectedstep(problem, subprobleminput, options, storedb, key)
% Helper function that uses cache when step rejected in trustregions.m.
% 
% The next step can be computed from information already computed in the 
% previous trs_tCG_cached loop which is stored accordingly. This function
% is only called when caching is used by trs_tCG_cached.
%
% Note there can only be two situations.
% 1. We return the same eta, Heta as the previous tCG loop and decrease
% Delta again (either d_Hd <= 0 or we use store_last)
% 2. We return a new eta when some previous candidate eta (stored in 
% store_iters) satisfies normsq := <eta,eta>_x >= Delta^2 at the current 
% Delta (exceeding trust region).
%
% See also: trustregions trs_tCG_cached trs_tCG

% This file is part of Manopt: www.manopt.org.
% Original author: Victor Liao, Jun. 24, 2022.

    x = subprobleminput.x;
    Delta = subprobleminput.Delta;

    lincomb = @(a, u, b, v) problem.M.lincomb(x, a, u, b, v);

    store = storedb.get(key);
    store_iters = store.store_iters;
    store_last = store.store_last;

    % get amount of memory that is currently cached.
    savedoutput.memorytCG_MB = getsize(store_iters(1))/1024^2 * length(store_iters) + getsize(store_last)/1024^2;

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
            
            savedoutput.limitedbyTR = true;
            savedoutput.numinner = store_iters(i).numinner;
            
            if options.verbosity == 2
                if options.useCache
                    print_str = sprintf('numinner: %5d     numstored: %d     %s', savedoutput.numinner, length(store_iters), stopreason_str);
                else
                    print_str = sprintf('numinner: %5d                       %s', savedoutput.numinner, stopreason_str);
                end
            elseif options.verbosity > 2
                if options.useCache
                    print_str = sprintf('\nnuminner: %5d     numstored: %d     memorytCG: %e[MB]     %s', savedoutput.numinner, length(store_iters), savedoutput.memorytCG_MB, stopreason_str);
                else
                    print_str = sprintf('\nnuminner: %5d                                             %s', savedoutput.numinner, stopreason_str);
                end
            end

            return;
        end
    end

    % If no struct in store_iters satisfies negative curvature or 
    % trust-region radius violation we must exit with last eta, Heta and
    % limitedbyTR = false.
    eta = store_last.eta;
    Heta = store_last.Heta;
    savedoutput.limitedbyTR = false;
    savedoutput.numinner = store_last.numinner;
    if options.verbosity == 2
        if options.useCache
            print_str = sprintf('numinner: %5d     numstored: %d     %s', savedoutput.numinner, length(store_iters), store_last.stopreason_str);
        else
            print_str = sprintf('numinner: %5d     %s', savedoutput.numinner, store_last.stopreason_str);
        end
    elseif options.verbosity > 2
        if options.useCache
            print_str = sprintf('\nnuminner: %5d     numstored: %d     memorytCG: %e[MB]     %s', savedoutput.numinner, length(store_iters), savedoutput.memorytCG_MB, store_last.stopreason_str);
        else
            print_str = sprintf('\nnuminner: %5d     %s', savedoutput.numinner, store_last.stopreason_str);
        end
    end
