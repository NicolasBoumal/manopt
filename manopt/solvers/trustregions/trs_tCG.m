function trsoutput = trs_tCG(problem, trsinput, options, storedb, key)
% Truncated (Steihaug-Toint) Conjugate-Gradient method.
%
% minimize <eta, fgradx> + .5*<eta, Hess(eta)>
% subject to <eta,eta>_[inverse precon] <= Delta^2
%
% function trsoutput = trs_tCG(problem, trsinput, options, storedb, key)
%
% The preconditioner (if one is available in the problem structure) is used
% exactly if the initial step eta0 (see below) is the zero vector.
%
% Inputs:
%   problem: Manopt optimization problem structure
%   trsinput: structure with the following fields:
%       x: point on the manifold problem.M
%       fgradx: tangent vector at x (typically, the gradient of the cost)
%       Delta: trust-region radius
%       eta0: initial tangent vector from where to run tCG (optional).
%             If it is unspecified, or [], or larger than Delta, or if
%             options.useRand is set to true, then eta0 is ignored. If
%             useRand is true, then a small random tangent vector is used
%             for eta0. Otherwise, eta0 defaults to the zero vector.
%   options: structure containing options for the subproblem solver
%   storedb, key: manopt's caching system for the point x
%
% Options specific to this subproblem solver:
%   kappa (0.1)
%       kappa convergence tolerance.
%       kappa > 0 is the linear convergence target rate: trs_tCG will
%       terminate early if the residual was reduced by a factor of kappa.
%   theta (1.0)
%       theta convergence tolerance.
%       1+theta (theta between 0 and 1) is the superlinear convergence
%       target rate. trs_tCG will terminate early if the residual was 
%       reduced by a power of 1+theta.
%   mininner (1)
%       Minimum number of inner iterations for trs_tCG.
%   maxinner (problem.M.dim() : the manifold's dimension)
%       Maximum number of inner iterations for trs_tCG.
%   useRand (false)
%       Set to true if the subproblem solver is to be initiated with a
%       random tangent vector. If set to true, no preconditioner will be
%       used. This option is set to true in some scenarios to escape saddle
%       points, but is otherwise seldom activated. 
%       
% Output: the structure trsoutput contains the following fields:
%   eta: approximate solution to the trust-region subproblem at x
%   Heta: Hess f(x)[eta] -- this is necessary in the outer loop, and it
%       is often naturally available to the subproblem solver at the
%       end of execution, so that it may be cheaper to return it here.
%   limitedbyTR: true if a boundary solution is returned
%   printstr: logged information to be printed by trustregions.
%   stats: structure with the following statistics:
%       numinner: number of inner loops before returning
%       hessvecevals: number of Hessian calls issued
%       cauchy: true if cauchy step was used. false by default. Will
%       be present in stats only if options.useRand = true
%
% trs_tCG can also be called in the following way (by trustregions) to
% obtain part of the header to print and an initial stats structure:
%
% function trsoutput = trs_tCG([], [], options)
%
% In this case trsoutput contains the following fields:
%   printheader: subproblem header to be printed before the first loop of 
%       trustregions
%   initstats: struct with initial values for stored stats in subsequent
%       calls to trs_tCG. Used in the first call to savestats 
%       in trustregions to initialize the info struct properly.
%
% See also: trustregions trs_tCG_cached trs_gep

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
%       tCG now also returns Heta, the Hessian at x along eta. Additional
%       esthetic modifications.
%
%   NB Dec. 2, 2013:
%       If options.useRand is activated, we now make sure the preconditio-
%       ner is not used, as was originally intended in GenRTR. In time, we
%       may want to investigate whether useRand can be modified to work
%       well with preconditioning too.
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
%
%   VL June 29, 2022:
%       Renamed tCG to trs_tCG to keep consistent naming with new
%       subproblem solvers for trustregions. Also modified inputs and 
%       outputs to be compatible with other subproblem solvers.
%
%   BS May 12, 2023:
%       Corrected update rule for e_Pd when useRand = true.
%
%  NB Nov. 17, 2023:
%       Added option trsinput.eta0 to allow specific nonzero initial vector
%       and modified the logic surrounding the choice of initial eta
%       accordingly. In particular, introduced the flag eta0_is_zero that
%       is now used explicitly instead of checking ~options.useRand as
%       being the determinant factor to know whether we apply
%       preconditioning or not. Moreover, in the early stopping criterion
%       based on small residual, we are now comparing against the norm of
%       trsinput.fgradx rather than the norm of the initial residual. This
%       is equivalent when eta0 = 0 (which was the default behavior), but
%       it is different when eta0 is not zero. The difference is only very
%       small for eta0 produced by useRand though, so it's almost
%       inconsequential as a change in behavior, and it's arguably the
%       right thing to do for general eta0.
%
%   NB Sep.  6, 2024:
%       Added output trsoutput.numericaltrouble as a detector of numerical
%       issues that may be used by trustregions to decide to stop early.

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
% if eta_0 = 0.
%
% [CGT2000] Conn, Gould and Toint: Trust-region methods, 2000.

% trustregions only wants header and default values for stats.
if nargin == 3 && isempty(problem) && isempty(trsinput)
    if isfield(options, 'useRand') && options.useRand
        trsoutput.printheader = sprintf('%9s   %9s   %11s   %s', ...
                                'numinner', 'hessvec', 'used_cauchy', ...
                                'stopreason');
        trsoutput.initstats = struct('numinner', 0, 'hessvecevals', 0, ...
                                'cauchy', false);
    else
        trsoutput.printheader = sprintf('%9s   %9s   %s', 'numinner', ...
                                'hessvec', 'stopreason');
        trsoutput.initstats = struct('numinner', 0, 'hessvecevals', 0);
    end
    return;
end

x = trsinput.x;
Delta = trsinput.Delta;
grad = trsinput.fgradx;
if isfield(trsinput, 'eta0')
    eta0 = trsinput.eta0;
else
    eta0 = [];
end

M = problem.M;

inner   = @(u, v) M.inner(x, u, v);
lincomb = @(a, u, b, v) M.lincomb(x, a, u, b, v);
tangent = @(u) M.tangent(x, u);

% Set local defaults here
localdefaults.kappa = 0.1;
localdefaults.theta = 1.0;
localdefaults.mininner = 1;
localdefaults.maxinner = M.dim();
localdefaults.useRand = false;

% Merge local defaults with user options, if any
if ~exist('options', 'var') || isempty(options)
    options = struct();
end
options = mergeOptions(localdefaults, options);

theta = options.theta;
kappa = options.kappa;

% This boolean is part of the outputs. It is set to true if the solution
% eta we return was limited in norm by the trust-region radius Delta.
limitedbyTR = false;

% Flip this to true if we encountered difficulties due to inexact
% arithmetic: trustregions may want to terminate if so.
numericaltrouble = false;

% If eta0 is provided, check whether we can use it, or if we ignore it.
if options.useRand && ~isempty(eta0)
    warning('manopt:tCG:eta0', ...
            ['Since options.useRand = true, trsinput.eta0 is ignored. ' ...
             'Set trsinput.eta0 = []; or options.useRand = false.']);
    eta0 = [];  % we will use a random eta0
end
if ~options.useRand && ~isempty(eta0) && M.norm(x, eta0) > Delta
    warning('manopt:tCG:eta0', ...
            ['trsinput.eta0 is ignored because it is larger than the ' ...
             'trust-region radius Delta. Set trsinput.eta0 = [];']);
    eta0 = [];  % we will use eta0 = 0
end

% This flag is used for initialization. Also, the preconditioner (if any)
% is only used if eta0 is zero: we use this flag to determine that.
% Note: we do not check whether (M.norm(x, eta0) == 0), because (a) that
% would cost a bit in some cases, and (b) it would create a hard-to-debug
% behavior where the preconditioner might be used on the off-chance that
% the eta0 supplied happens to be a zero vector. Thus, if the user wants a
% zero vector "on purpose", they should set eta0 to [], which is the
% default.
eta0_is_zero = ~options.useRand && isempty(eta0);

if eta0_is_zero
    eta = M.zerovec(x);
    Heta = M.zerovec(x);
    r = grad;
    e_Pe = 0;
else
    if options.useRand
        % Random vector in T_x M (this has to be very small, inside TR)
        eta = M.randvec(x);
        eta_norm = M.norm(x, eta);
        eta = M.lincomb(x, min(2^(-20), 2^(-13)*Delta/eta_norm), eta);
    else
        eta = eta0;
    end
    Heta = getHessian(problem, x, eta, storedb, key);
    r = lincomb(1, grad, 1, Heta);
    e_Pe = inner(eta, eta);
end

r_r = inner(r, r);
norm_r = sqrt(r_r);
if eta0_is_zero
    norm_g = norm_r;
else
    norm_g = M.norm(x, grad);
end

% Precondition the residual if eta0 = 0.
if eta0_is_zero
    z = getPrecon(problem, x, r, storedb, key);
else
    z = r;
end

% Compute z'*r.
z_r = inner(z, r);
d_Pd = z_r;

% Initial search direction (we maintain -delta in memory, called mdelta, to
% avoid a change of sign of the tangent vector.)
mdelta = z;
if eta0_is_zero
    e_Pd = 0;
else % and therefore, no preconditioner
    e_Pd = -inner(eta, mdelta);
end

% If the Hessian or a linear Hessian approximation is in use, it is
% theoretically guaranteed that the model value decreases strictly
% with each iteration of tCG. Hence, there is no need to monitor the model
% value. But, when a nonlinear Hessian approximation is used (such as the
% built-in finite-difference approximation for example), the model may
% increase. It is then important to terminate the tCG iterations and return
% the previous (the best-so-far) iterate. The variable below will hold the
% model value.
%
% This computation could be further improved based on Section 17.4.1 in
% Conn, Gould, Toint, Trust Region Methods, 2000.
% If we make this change, then also modify trustregions to gather this
% value from tCG rather than recomputing it itself.
model_fun = @(eta, Heta) inner(eta, grad) + .5*inner(eta, Heta);
if eta0_is_zero
    model_value = 0;
else
    model_value = model_fun(eta, Heta);
end

% Pre-assume termination because j == end.
stopreason_str = 'maximum inner iterations';

% Begin inner/tCG loop.
for j = 1 : options.maxinner
    
    % This call is the computationally expensive step.
    Hmdelta = getHessian(problem, x, mdelta, storedb, key);
    
    % Compute curvature.
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
        tau = (-e_Pd + sqrt(e_Pd*e_Pd + d_Pd*(Delta^2-e_Pe))) / d_Pd;
        if options.debug > 2
            fprintf('DBG:     tau  : %e\n', tau);
        end
        eta = lincomb(1,  eta, -tau,  mdelta);
        
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
        stopreason_str = 'model increased in last inner iteration';
        numericaltrouble = true;
        break;
    end
    
    eta = new_eta;
    Heta = new_Heta;
    model_value = new_model_value;
    
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
    if j >= options.mininner && norm_r <= norm_g*min(norm_g^theta, kappa)
        % Residual is small enough to quit
        if kappa < norm_g^theta
            stopreason_str = 'reached target residual-kappa (linear)';
        else
            stopreason_str = 'reached target residual-theta (superlinear)';
        end
        break;
    end
    
    % Precondition the residual.
    if eta0_is_zero
        z = getPrecon(problem, x, r, storedb, key);
    else
        z = r;
    end
    
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
    if eta0_is_zero
        % This update rule is cheap, since it avoids the need for
        % a preconditioner call or an inner product: it is all scalars.
        e_Pd = beta*(e_Pd + alpha*d_Pd);
    else
        % When eta0 is not zero, then the update rule above is not exact.
        % (It is still approximately if eta0 is small.)
        % To avoid problems, we simply compute the inner product directly.
        % This is less of an issue here, since there is no preconditioner.
        e_Pd = -inner(eta, mdelta);
    end
    d_Pd = z_r + beta*beta*d_Pd;
    
end  % of tCG loop

% If using randomized approach, compare result with the Cauchy point.
% Convergence proofs assume that we achieve at least (a fraction of)
% the reduction of the Cauchy point. After this if-block, either all
% eta-related quantities have been changed consistently, or none of
% them have changed.
if options.useRand
    used_cauchy = false;
    
    % Check the curvature,
    Hg = getHessian(problem, x, grad, storedb, key);
    g_Hg = M.inner(x, grad, Hg);
    if g_Hg <= 0
        tau_c = 1;
    else
        tau_c = min( norm_g^3/(Delta*g_Hg) , 1);
    end
    % and generate the Cauchy point.
    eta_c  = M.lincomb(x, -tau_c * Delta / norm_g, grad);
    Heta_c = M.lincomb(x, -tau_c * Delta / norm_g, Hg);

    % Now that we have computed the Cauchy point in addition to the
    % would-be returned eta, we might as well keep the best of them.
    mdle  = M.inner(x, grad, eta) + .5*M.inner(x, Heta, eta);
    mdlec = M.inner(x, grad, eta_c) + .5*M.inner(x, Heta_c, eta_c);

    if mdlec < mdle
        eta = eta_c;
        Heta = Heta_c;
        used_cauchy = true;
    end
end

printstr = sprintf('%9d   %9d   %s', j, j, stopreason_str);
stats = struct('numinner', j, 'hessvecevals', j);

if options.useRand
    stats.cauchy = used_cauchy;
    printstr = sprintf('%9d   %9d   %11s   %s', j, j, ...
                string(used_cauchy), stopreason_str);
end

trsoutput.eta = eta;
trsoutput.Heta = Heta;
trsoutput.limitedbyTR = limitedbyTR;
trsoutput.numericaltrouble = numericaltrouble;
trsoutput.printstr = printstr;
trsoutput.stats = stats;
end
