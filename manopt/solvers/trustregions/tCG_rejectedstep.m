function output = tCG_rejectedstep(problem, subprobleminput, options, storedb, key)
% Helper function to handle when a step is rejected in
% trustregions.m rather than the tCG loop. The next step can be computed from 
% information already computed in the previous trs_tCG call which is stored
% accordingly.

% Note there can only be two situations.
% 1. We return the same eta, Heta as the previous tCG call and decrease
% Delta again (either d_Hd <= 0 or we use store_last)
% 2. We return a new eta when some previous candidate eta (stored in 
% store_iters) satisfies normsq:=<eta,eta>_x >= Delta^2 at the current 
% Delta (exceeding trust region).
%
% See also: trustregions, trs_tCG_cached, trs_tCG_legacy

% This file is part of Manopt: www.manopt.org.
% Original author: Victor Liao, Jun. 24, 2022.

    x = subprobleminput.x;
    eta = subprobleminput.eta;
    Delta = subprobleminput.Delta;

    lincomb = @(a, u, b, v) problem.M.lincomb(x, a, u, b, v);

    store = storedb.get(key);
    store_iters = store.store_iters;
    store_last = store.store_last;

    % getSize for one entry in store_iters which will be the same for
    % all others.
    perIterMemory_MB = getSize(store_iters(1))/1024^2;

    memorytCG_MB = perIterMemory_MB * length(store_iters) + getSize(store_last)/1024^2;
    
    if memorytCG_MB > options.memorytCG_warningval
        warning('manopt:trs_tCG_cached:memory', ...
        [sprintf('trs_tCG_cached will cache %.2f [MB] for at least one iteration of trustregions until a step is accepted.', memorytCG_MB) ...
        'If memory is limited turn off caching by options.useCache = false.\n' ...
        'To disable this warning: warning(''off'', ''manopt:trs_tCG_cached:memory'')']);
    end

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
            numit = store_iters(i).numit;
            break;
        elseif i == length(store_iters)
            % We exit with last eta, Heta
            output = store_last;
            output.memorytCG_MB = memorytCG_MB;
            output.limitedbyTR = false;
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
        output.stopreason_str = 'negative curvature';
    else
        output.stopreason_str = 'exceeded trust region';
    end
    output.eta = eta;
    output.Heta = Heta;
    output.numit = numit;
    output.limitedbyTR = true;
    output.memorytCG_MB = memorytCG_MB;
