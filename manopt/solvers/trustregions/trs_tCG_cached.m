function output = trs_tCG_cached(problem, subprobleminput, options, storedb, key)
% trs_tCG_cached - Truncated (Steihaug-Toint) Conjugate-Gradient method where
% information is stored in case the step is rejected by trustregions.m to
% compute the next step. If the previous step is rejected 
% work is passed to tCG_rejectedstep.m to process the stored information.

% minimize <eta,grad> + .5*<eta,Hess(eta)>
% subject to <eta,eta>_[inverse precon] <= Delta^2
%
% See also: trustregions

% This file is part of Manopt: www.manopt.org.
% This code is an adaptation to Manopt of the original GenRTR code:
% RTR - Riemannian Trust-Region
% (c) 2004-2007, P.-A. Absil, C. G. Baker, K. A. Gallivan
% Florida State University
% School of Computational Science
% (http://www.math.fsu.edu/~cbaker/GenRTR/?page=download)
% See accompanying license file.
% The adaptation was executed by Nicolas Boumal.
%
% Change log:
%
%   NB Feb. 12, 2013:
%       We do not project r back to the tangent space anymore: it was not
%       necessary, and as of Manopt 1.0.1, the proj operator does not
%       coincide with this notion anymore.
%
%   NB April 3, 2013:
%       trs_tCG now also returns Heta, the Hessian at x along eta. Additional
%       esthetic modifications.
%
%   NB Dec. 2, 2013:
%       If options.useRand is activated, we now make sure the preconditio-
%       ner is not used, as was originally intended in GenRTR. In time, we
%       may want to investigate whether useRand can be modified to work well
%       with preconditioning too.
%
%   NB Jan. 9, 2014:
%       Now checking explicitly for model decrease at each iteration. The
%       first iteration is a Cauchy point, which necessarily realizes a
%       decrease of the model cost. If a model increase is witnessed
%       (which is theoretically impossible if a linear operator is used for
%       the Hessian approximation), then we return the previous eta. This
%       ensures we always achieve at least the Cauchy decrease, which
%       should be sufficient for convergence.
%
%   NB Feb. 17, 2015:
%       The previous update was in effect verifying that the current eta
%       performed at least as well as the first eta (the Cauchy step) with
%       respect to the model cost. While this is an acceptable strategy,
%       the documentation (and the original intent) was to ensure a
%       monotonic decrease of the model cost at each new eta. This is now
%       the case, with the added line: "model_value = new_model_value;".
%
%   NB April 3, 2015:
%       Works with the new StoreDB class system.
%
%   NB April 17, 2018:
%       Two changes:
%        (a) Instead of updating delta and Hdelta, we now update -delta and
%            -Hdelta, named mdelta and Hmdelta. This allows to spare one
%            call to lincomb(x, -1, z).
%        (b) We re-project mdelta to the tangent space at every iteration,
%            to avoid drifting away from it. The choice to project mdelta
%            specifically is motivated by the fact that this is the vector
%            passed to getHessian, hence this is where accurate tangency
%            may be most important. (All other operations are linear
%            combinations of tangent vectors, which should be fairly safe.)
%   VL June 24, 2022:
%       Now, we store information at each
%       iteration in trs_tCG_cached compared to trs_tCG. This may be useful 
%       for the next call to trs_tCG_cached and the work is passed to 
%       tCG_rejectedstep rather than the normal trs_tCG loop.

% All terms involving the trust-region radius use an inner product
% w.r.t. the preconditioner; this is because the iterates grow in
% length w.r.t. the preconditioner, guaranteeing that we do not
% re-enter the trust-region.
%
% The following recurrences for Prec-based norms and inner
% products come from [CGT2000], pg. 205, first edition.
% Below, P is the preconditioner.
%
% <eta_k,P*delta_k> = 
%          beta_k-1 * ( <eta_k-1,P*delta_k-1> + alpha_k-1 |delta_k-1|^2_P )
% |delta_k|^2_P = <r_k,z_k> + beta_k-1^2 |delta_k-1|^2_P
%
% Therefore, we need to keep track of:
% 1)   |delta_k|^2_P
% 2)   <eta_k,P*delta_k> = <eta_k,delta_k>_P
% 3)   |eta_k  |^2_P
%
% Initial values are given by
%    |delta_0|_P = <r,z>
%    |eta_0|_P   = 0
%    <eta_0,delta_0>_P = 0
% because we take eta_0 = 0 (if useRand = false).
%
% [CGT2000] Conn, Gould and Toint: Trust-region methods, 2000.

if options.useRand
    fprintf('options.useRand = true but trs_tCG_cached.m does not use this option! trs_tCG will be used instead. \n');
    output = trs_tCG(problem, subprobleminput, options, storedb, key);
    return;
end

% Previous step was rejected so we can save some compute here by passing to
% helper function.
if ~subprobleminput.accept
        output = tCG_rejectedstep(problem, subprobleminput, options, storedb, key);
        return;
end

x = subprobleminput.x;
eta = subprobleminput.eta;
Delta = subprobleminput.Delta;
grad = subprobleminput.fgradx;

inner   = @(u, v) problem.M.inner(x, u, v);
lincomb = @(a, u, b, v) problem.M.lincomb(x, a, u, b, v);
tangent = @(u) problem.M.tangent(x, u);

% returned boolean to trustregions.m. true if we are limited by the TR
% boundary (returns boundary solution). Otherwise false.
limitedbyTR = false;

theta = options.theta;
kappa = options.kappa;

Heta = problem.M.zerovec(x);
r = grad;
e_Pe = 0;

r_r = inner(r, r);
norm_r = sqrt(r_r);
norm_r0 = norm_r;

% Precondition the residual.
z = getPrecon(problem, x, r, storedb, key);

% Compute z'*r.
z_r = inner(z, r);
d_Pd = z_r;

% Initial search direction (we maintain -delta in memory, called mdelta, to
% avoid a change of sign of the tangent vector.)
mdelta = z;
e_Pd = 0;

% If the Hessian or a linear Hessian approximation is in use, it is
% theoretically guaranteed that the model value decreases strictly
% with each iteration of trs_tCG. Hence, there is no need to monitor the model
% value. But, when a nonlinear Hessian approximation is used (such as the
% built-in finite-difference approximation for example), the model may
% increase. It is then important to terminate the trs_tCG iterations and return
% the previous (the best-so-far) iterate. The variable below will hold the
% model value.
%
% This computation could be further improved based on Section 17.4.1 in
% Conn, Gould, Toint, Trust Region Methods, 2000.
% If we make this change, then also modify trustregions to gather this
% value from trs_tCG rather than recomputing it itself.
model_fun = @(eta, Heta) inner(eta, grad) + .5*inner(eta, Heta);
model_value = 0;

% Pre-assume termination because j == end.
stopreason_str = 'maximum inner iterations';

% Track certain iterations in case step is rejected.
% store_iters tracks candidate etas with increasing squared
% norm relevant when limitedbyTR = true, or when <eta, Heta> <= 0
store_iters = struct('normsq', [], 'numit', [], 'e_Pe', [], ...
    'd_Pd', [], 'e_Pd', [], 'd_Hd', [], 'eta', [], 'Heta', [], ...
    'mdelta', [], 'Hmdelta', []);

% If exit is prompted when limitedbyTR = false then we store different info
store_last = struct('numit', [], 'stopreason_str', [], 'eta', [], 'Heta', []);

store_index = 1;
max_normsq = 0;

% only need to compute memory for one item in store_iters in Megabytes(MB)
perIterMemory_MB = 0;

% total cached memory stored in MB
memorytCG_MB = 0;

% Begin inner/trs_tCG loop.
for j = 1 : options.maxinner
    
    % This call is the computationally expensive step.
    Hmdelta = getHessian(problem, x, mdelta, storedb, key);
    
    % Compute curvature (often called kappa).
    d_Hd = inner(mdelta, Hmdelta);
    
    
    % Note that if d_Hd == 0, we will exit at the next "if" anyway.
    alpha = z_r/d_Hd;
    % <neweta,neweta>_P =
    % <eta,eta>_P + 2*alpha*<eta,delta>_P + alpha*alpha*<delta,delta>_P
    e_Pe_new = e_Pe + 2.0*alpha*e_Pd + alpha*alpha*d_Pd;
    
    if options.debug > 2
        fprintf('DBG:   (r,r)  : %e\n', r_r);
        fprintf('DBG:   (d,Hd) : %e\n', d_Hd);
        fprintf('DBG:   alpha  : %e\n', alpha);
    end

    % Selectively store info in store_iter.
    % next_smallest = ((1/4)^(n) Delta)^2 where n is the smallest integer
    % such that max_normsq <= next_smallest. 
    % We use this condition to only store relevant iterations in case of
    % rejection in trustregions.m
    if max_normsq > 0
        next_smallest = (1/16)^floor(-(1/4)*(log2(max_normsq) - ...
                                            log2(Delta^2))) * Delta^2;
    else
        next_smallest = 0;
    end

    if d_Hd <= 0 || e_Pe_new >= next_smallest
        store_iters(store_index) = struct('normsq', e_Pe_new, 'numit', j,...
                         'e_Pe', e_Pe, 'd_Pd', d_Pd, 'e_Pd', e_Pd,...
                         'd_Hd', d_Hd, 'eta', eta, 'Heta', Heta, ...
                         'mdelta', mdelta, 'Hmdelta', Hmdelta);
        max_normsq = e_Pe_new;

        % getSize for one entry in store_iters which will be the same for
        % all others.
        if perIterMemory_MB == 0
            perIterMemory_MB = getSize(store_iters(store_index))/1024^2;
        end

        memorytCG_MB = memorytCG_MB + perIterMemory_MB;
        
        if memorytCG_MB > options.memorytCG_warningnum
            warning('manopt:trs_tCG_cached:memory', ...
            [sprintf('trs_tCG_cached will cache %.2f [MB]. If memory ', memorytCG_MB) ...
            'is limited turn off caching.\n' ...
            'To disable this warning: warning(''off'', ''manopt:trs_tCG_cached:memory'')']);
        end
        
        store_index = store_index + 1;
    end

    % Check against negative curvature and trust-region radius violation.
    % If either condition triggers, we bail out.
    if d_Hd <= 0 || e_Pe_new >= Delta^2
        % want
        %  ee = <eta,eta>_prec,x
        %  ed = <eta,delta>_prec,x
        %  dd = <delta,delta>_prec,x
        % Note (Nov. 26, 2021, NB): numerically, it might be better to call
        %   tau = max(real(roots([d_Pd, 2*e_Pd, e_Pe-Delta^2])));
        % This should be checked.
        % Also, we should safe-guard against 0/0: could happen if grad = 0.

        % store new struct containing all the required info in store_iter
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
        
        limitedbyTR = true;
        
        if d_Hd <= 0
            stopreason_str = 'negative curvature';
        else
            stopreason_str = 'exceeded trust region';
        end
        break;
    end
    
    % No negative curvature and eta_prop inside TR: accept it.
    e_Pe = e_Pe_new;
    new_eta  = lincomb(1,  eta, -alpha,  mdelta);
    
    % If only a nonlinear Hessian approximation is available, this is
    % only approximately correct, but saves an additional Hessian call.
    % TODO: this computation is redundant with that of r, L241. Clean up.
    new_Heta = lincomb(1, Heta, -alpha, Hmdelta);
    
    % Verify that the model cost decreased in going from eta to new_eta. If
    % it did not (which can only occur if the Hessian approximation is
    % nonlinear or because of numerical errors), then we return the
    % previous eta (which necessarily is the best reached so far, according
    % to the model cost). Otherwise, we accept the new eta and go on.
    new_model_value = model_fun(new_eta, new_Heta);
    if new_model_value >= model_value
        stopreason_str = 'model increased';
        break;
    end
    
    eta = new_eta;
    Heta = new_Heta;
    model_value = new_model_value; %% added Feb. 17, 2015
    
    % Update the residual.
    r = lincomb(1, r, -alpha, Hmdelta);
    
    % Compute new norm of r.
    r_r = inner(r, r);
    norm_r = sqrt(r_r);

    % Check kappa/theta stopping criterion.
    % Note that it is somewhat arbitrary whether to check this stopping
    % criterion on the r's (the gradients) or on the z's (the
    % preconditioned gradients). [CGT2000], page 206, mentions both as
    % acceptable criteria.
    if j >= options.mininner && norm_r <= norm_r0*min(norm_r0^theta, kappa)
        % Residual is small enough to quit
        if kappa < norm_r0^theta
            stopreason_str = 'reached target residual-kappa (linear)';
        else
            stopreason_str = 'reached target residual-theta (superlinear)';
        end
        break;
    end
    
    % Precondition the residual.
    z = getPrecon(problem, x, r, storedb, key);
    
    % Save the old z'*r.
    zold_rold = z_r;
    % Compute new z'*r.
    z_r = inner(z, r);
    
    % Compute new search direction.
    beta = z_r/zold_rold;
    mdelta = lincomb(1, z, beta, mdelta);
    
    % Since mdelta is passed to getHessian, which is the part of the code
    % we have least control over from here, we want to make sure mdelta is
    % a tangent vector up to numerical errors that should remain small.
    % For this reason, we re-project mdelta to the tangent space.
    % In limited tests, it was observed that it is a good idea to project
    % at every iteration rather than only every k iterations, the reason
    % being that loss of tangency can lead to more inner iterations being
    % run, which leads to an overall higher computational cost.
    mdelta = tangent(mdelta);
    
    % Update new P-norms and P-dots [CGT2000, eq. 7.5.6 & 7.5.7].
    e_Pd = beta*(e_Pd + alpha*d_Pd);
    d_Pd = z_r + beta*beta*d_Pd;
    
end  % of trs_tCG loop

% Store in case we did not exit because we were limited by TR (model value 
% increased or kappa/theta stopping criterion satisfied)
if ~limitedbyTR
    store_last = struct('numit', j, 'stopreason_str', ...
        stopreason_str, 'eta', eta, 'Heta', Heta);
    memorytCG_MB = memorytCG_MB + getSize(store_last)/1024^2;
    
    if memorytCG_MB > options.memorytCG_warningnum
            warning('manopt:trs_tCG_cached:memory', ...
            [sprintf('trs_tCG_cached will cache %.2f [MB]. If memory ', memorytCG_MB) ...
            'is limited turn off caching.\n' ...
            'To disable this warning: warning(''off'', ''manopt:trs_tCG_cached:memory'')']);
    end
end

store = storedb.get(key);
store.store_iters = store_iters;
store.store_last = store_last;
storedb.set(store, key);

output.eta = eta;
output.Heta = Heta;
output.numit = j;
output.stopreason_str = stopreason_str;
output.limitedbyTR = limitedbyTR;
output.memorytCG_MB = memorytCG_MB;
end
