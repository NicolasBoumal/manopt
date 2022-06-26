function [eta, Heta, inner_it, stop_tCG] ...
                 = tCG_rejectedstep(problem, x, eta, Delta, options, storedb, key)
% Helper function to handle when a step is rejected in
% trustregions.m rather than the tCG loop. The next step can be computed from 
% information already computed in the previous tCG call which is stored
% accordingly.

% Note there can only be two situations.
% 1. We return the same eta, Heta as the previous tCG call and decrease
% Delta again (either d_Hd <= 0 or we use store_last)
% 2. We return a new eta when some previous candidate eta (stored in 
% store_iters) satisfies normsq:=<eta,eta>_x >= Delta^2 at the current 
% Delta (exceeding trust region).
%
% See also: trustregions, tCG_efficient, tCG

% This file is part of Manopt: www.manopt.org.
% Original author: Victor Liao, Jun. 24, 2022.

        lincomb = @(a, u, b, v) problem.M.lincomb(x, a, u, b, v);

        store = storedb.getWithShared(key);
        store_iters = store.store_iters;
        store_last = store.store_last;

        for i=1:length(store_iters)
            normsq = store_iters(i).normsq;
            d_Hd = store_iters(i).d_Hd;
            if d_Hd <= 0 || normsq >= Delta^2
                % We exit but need to compute eta, Heta dependent on Delta
                e_Pe = store_iters(i).e_Pe;
                e_Pd = store_iters(i).e_Pd;
                d_Pd = store_iters(i).d_Pd;
                eta = store_iters(i).eta;
                mdelta = store_iters(i).mdelta;
                Hmdelta = store_iters(i).Hmdelta;
                Heta = store_iters(i).Heta;
                inner_it = store_iters(i).inner_it;
                break;
            elseif i == length(store_iters)
                % We exit with last eta, Heta
                eta = store_last.eta;
                Heta = store_last.Heta;
                inner_it = store_last.inner_it;
                stop_tCG = store_last.stop_tCG;
                return;
            end
        end

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
            stop_tCG = 1;     % negative curvature
        else
            stop_tCG = 2;     % exceeded trust region
        end
