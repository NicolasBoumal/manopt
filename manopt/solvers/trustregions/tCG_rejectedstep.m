function [eta, Heta, inner_it, stop_tCG] ...
                 = tCG_rejectedstep(problem, x, eta, Delta, options, storedb, key)
        % want
        %  ee = <eta,eta>_prec,x
        %  ed = <eta,delta>_prec,x
        %  dd = <delta,delta>_prec,x
        % Note (Nov. 26, 2021, NB): numerically, it might be better to call
        %   tau = max(real(roots([d_Pd, 2*e_Pd, e_Pe-Delta^2])));
        % This should be checked.
        % Also, we should safe-guard against 0/0: could happen if grad = 0.
        inner   = @(u, v) problem.M.inner(x, u, v);
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
        
        Hmdelta = getHessian(problem, x, mdelta, storedb, key);
        
        d_Hd = inner(mdelta, Hmdelta);

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
