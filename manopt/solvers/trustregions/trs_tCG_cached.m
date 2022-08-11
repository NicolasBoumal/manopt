function [eta, Heta, print_str, stats] = trs_tCG_cached(problem, subprobleminput, options, storedb, key)
% Truncated (Steihaug-Toint) Conjugate-Gradient method with caching.
%
% minimize <eta,grad> + .5*<eta,Hess(eta)>
% subject to <eta,eta>_[inverse precon] <= Delta^2
%
% function [~, ~, ~, stats] = trs_tCG_cached()
% function [eta, Heta, print_str, stats] = trs_tCG_cached(problem, subprobleminput, options, storedb, key)
%
% trs_tCG_cached by default stores information (when options.useCache=true) 
% which can help avoid redundant computations (using tCG_rejectedstep) 
% upon step rejection by trustregions.m.
%
% In contrast, trs_tCG does not cache so the redundant steps will
% still be computed but no extra memory is needed compared to
% trs_tCG_cached.
%
% trs_tCG_cached also cannot use randomization (so options.useRand should 
% be false).
%
% Inputs:
%   problem: Manopt optimization problem structure
%   subprobleminput: struct storing information for this subproblemsolver
%       x: point on the manifold problem.M
%       grad: gradient of the cost function of the problem at x
%       eta: starting point (normally problem.M.zerovec(x))
%       Delta = trust-region radius
%   options: structure containing options for the subproblem solver
%   storedb, key: caching data for problem at x
%
% Options specific to this subproblem solver:
%   kappa (0.1)
%       kappa convergence tolerance.
%       kappa > 0 is the linear convergence target rate: trs_tCG_cached will
%       terminate early if the residual was reduced by a factor of kappa.
%   theta (1.0)
%       theta convergence tolerance.
%       1+theta (theta between 0 and 1) is the superlinear convergence
%       target rate. trs_tCG_cached will terminate early if the residual 
%       was reduced by a power of 1+theta.
%   mininner (1)
%       Minimum number of inner iterations for trs_tCG_cached.
%   maxinner (problem.M.dim() : the manifold's dimension)
%       Maximum number of inner iterations for trs_tCG_cached.
%   useCache (true)
%       Set to false if no caching for the trs_tCG_cached is desired. It is
%       default true to improve computation time if there are many step
%       rejections in trustregions.m. Setting useCache to false can reduce
%       memory usage.
%   memorytCG_warningtol (1000)
%       Tolerance memory value in MB before issuing warning when 
%       useCache = true.
%       The default is 1GB but this value can be increased depending on the
%       user's machine, or to disable the warning completely use: 
%       warning(''off'', ''manopt:trs_tCG_cached:memory'')
%
% Outputs:
%   eta: approximate solution to the trust-region subproblem at x
%   Heta: Hess f(x)[eta] -- this is necessary in the outer loop, and it
%       is often naturally available to the subproblem solver at the
%       end of execution, so that it may be cheaper to return it here.
%   print_str: subproblem specific string to be printed by trustregions.m
%   stats: structure with values to be stored in trustregions.m
%       numinner: number of inner loops before returning
%       hessvecevals: number of Hessian calls during execution (can
%       differ from numinner when caching is used)
%       limitedbyTR: true if a boundary solution is returned
%       memorytCG_MB: memory of store_iters and store_last in MB
%
% Stored Information:
%   store_iters: a struct array with enough information to compute the next
%       step upon step rejection when the algorithm exits due to negative 
%       curvature or trust-region radius violation.
%   store_last: an additional struct to store_iters to compute the next
%       step upon step rejection when the algorithm exits but not due to 
%       negative curvature or trust-region radius violation
%
% Note: If nargin == 0, then the returned stats struct will contain the 
% relevant fields along with their corresponding initial values. print_str 
% will also contain the header to be printed before the first pass of 
% trustregions.m (if options.verbosity == 2). The other outputs will be 
% empty. This stats struct is used in the first call to savestats in 
% trustregions.m to initialize the info struct properly.
%
% See also: trustregions trs_tCG tCG_rejectedstep

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
%   VL June 24, 2022:
%       trs_tCG_cached by default stores information at each iteration
%       compared to trs_tCG.
%       This can be useful for the next call to trs_tCG_cached and the work 
%       is passed to tCG_rejectedstep rather than the normal tCG loop.

% See trs_tCG for references to relevant equations in
% [CGT2000] Conn, Gould and Toint: Trust-region methods, 2000.

if nargin == 0
    % trustregions.m only wants default values for stats.
    eta = [];
    Heta = [];
    print_str = sprintf('%-13s%-13s%-13s%s\n', 'numinner', 'hessvecevals', 'numstored', 'stopreason');
    stats = struct('numinner', 0, 'hessvecevals', 0, 'limitedbyTR', false, 'memorytCG_MB', 0);
    return;
end

if options.useRand
    warning('manopt:trs_tCG_cached:rand', ...
    [sprintf(['trs_tCG_cached does not use randomization. Did you mean to use trs_tCG?\n', ...
                'The initial eta will be a random small vector in T_x M but this may not be desired.\n',...
                'To silence this warning use randomization with trs_tCG or set options.useRand = false.']);]);
end

x = subprobleminput.x;
eta = subprobleminput.eta;
Delta = subprobleminput.Delta;
grad = subprobleminput.fgradx;

inner   = @(u, v) problem.M.inner(x, u, v);
lincomb = @(a, u, b, v) problem.M.lincomb(x, a, u, b, v);
tangent = @(u) problem.M.tangent(x, u);

% Set local defaults here
localdefaults.kappa = 0.1;
localdefaults.theta = 1.0;
localdefaults.mininner = 1;
localdefaults.maxinner = problem.M.dim();
localdefaults.useCache = true;
localdefaults.memorytCG_warningtol = 1000;

% Merge local defaults with user options, if any
if ~exist('options', 'var') || isempty(options)
    options = struct();
end
options = mergeOptions(localdefaults, options);

% Previous step was rejected so we can save some compute here by passing to
% helper function.
if ~subprobleminput.accept && options.useCache
        [eta, Heta, print_str, stats] = tCG_rejectedstep(problem, subprobleminput, options, storedb, key);
        return;
end

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
store_iters = struct('normsq', [], 'numinner', [], 'e_Pe', [], ...
    'd_Pd', [], 'e_Pd', [], 'd_Hd', [], 'eta', [], 'Heta', [], ...
    'mdelta', [], 'Hmdelta', []);

max_normsq = 0;

% only need to compute memory for one item in store_iters in Megabytes(MB)
peritermemory_MB = 0;

% total cached memory stored in MB
memorytCG_MB = 0;

% number of iterations where trs_tCG_cached stores information. This value
% will be length(store_iters) plus 1 if store_last is used.
numstored = 0;

% string that is printed by trustregions.m. For printing
% per-iteration information
print_str = '';

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

    if options.useCache
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
            numstored = numstored + 1;

            store_iters(numstored) = struct('normsq', e_Pe_new, 'numinner', ...
                             j, 'e_Pe', e_Pe, 'd_Pd', d_Pd, 'e_Pd', e_Pd,...
                             'd_Hd', d_Hd, 'eta', eta, 'Heta', Heta, ...
                             'mdelta', mdelta, 'Hmdelta', Hmdelta);
            max_normsq = e_Pe_new;
    
            % getSize for one entry in store_iters which will be the same for
            % all others.
            if peritermemory_MB == 0
                peritermemory_MB = getsize(store_iters(numstored))/1024^2;
            end
    
            memorytCG_MB = memorytCG_MB + peritermemory_MB;
            
            if memorytCG_MB > options.memorytCG_warningtol
                warning('manopt:trs_tCG_cached:memory', ...
                [sprintf('trs_tCG_cached will cache %.2f [MB] for at least one iteration of trustregions until a step is accepted.', memorytCG_MB) ...
                'If memory is limited turn off caching by options.useCache = false.\n' ...
                'To disable this warning: warning(''off'', ''manopt:trs_tCG_cached:memory'')']);
             end
            
        end
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

if options.useCache
    store = storedb.get(key);
    store.store_iters = store_iters;
    if ~limitedbyTR
        % Store extra information since we did not exit because we were 
        % limited by TR (model value increased or kappa/theta stopping 
        % criterion satisfied)
        store_last = struct('numinner', j, 'stopreason_str', ...
            stopreason_str, 'eta', eta, 'Heta', Heta);
        memorytCG_MB = memorytCG_MB + getsize(store_last)/1024^2;
        
        if memorytCG_MB > options.memorytCG_warningtol
            warning('manopt:trs_tCG_cached:memory', ...
            [sprintf('trs_tCG_cached will cache %.2f [MB] for at least one iteration of trustregions until a step is accepted.', memorytCG_MB) ...
            'If memory is limited turn off caching by options.useCache = false.\n' ...
            'If more memory can be used without problem increase options.memorytCG_warningtol accordingly.\n' ...
            'To disable this warning: warning(''off'', ''manopt:trs_tCG_cached:memory'')']);
        end
        store.store_last = store_last;
        
        numstored = numstored + 1;
    end

    storedb.set(store, key);
end

if options.verbosity == 2
    print_str = sprintf('    %-5d        %-5d        %-5d        %s', j, j, numstored, stopreason_str);
elseif options.verbosity == 3
    print_str = sprintf('numinner: %5d   hessvecevals: %5d   numstored: %5d   memorytCG: %8.2f[MB]   %s', j, j, numstored, memorytCG_MB, stopreason_str);
elseif options.verbosity > 3
    print_str = sprintf('\nnuminner: %5d   hessvecevals: %5d   numstored: %5d   memorytCG: %8.2f[MB]   %s', j, j, numstored, memorytCG_MB, stopreason_str);
end

stats = struct('numinner', j, 'hessvecevals', j, 'limitedbyTR', limitedbyTR, ...
                    'memorytCG_MB', memorytCG_MB);
end
