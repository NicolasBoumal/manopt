function trsoutput = trs_tCG_randomized(problem, trsinput, options, storedb, key)
% Modified truncated conjugate gradient method w/ randomized initial point.
%
% minimize <eta, fgradx> + .5*<eta, Hess(eta)>
% subject to <eta, eta> <= Delta^2
%
% function trsoutput = trs_tCG_randomized(problem, trsinput, options, storedb, key)
%
% This is meant to be used with trustregions_randomized.
%
% Inputs:
%   problem: Manopt optimization problem structure
%   trsinput: structure with the following fields:
%       x: point on the manifold problem.M
%       fgradx: tangent vector at x (typically, the gradient of the cost)
%       Delta: trust-region radius
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
%       Minimum number of inner iterations.
%   maxinner (problem.M.dim() : the manifold's dimension)
%       Maximum number of inner iterations.
%   hessianshift (0)
%       If nonzero, then the Hessian is replaced by
%           Hessian + hessianshift*identity.
%       A typical use would be to set hessianshift = sqrt(eps), to resolve
%       certain numerical issues.
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
%
% This can also be called in the following way (by trustregions) to
% obtain part of the header to print and an initial stats structure:
%
% function trsoutput = trs_tCG_randomized([], [], options)
%
% In this case trsoutput contains the following fields:
%   printheader: subproblem header to be printed before the first loop of 
%       trustregions.
%   initstats: struct with initial values for stored stats in subsequent
%       calls to this trs. Used in the first call to savestats 
%       in trustregions to initialize the info struct properly.
%
% See also: trustregions_randomized trustregions trs_tCG trs_tCG_cached

% This file is part of Manopt: www.manopt.org.
% Original authors: Nicolas Boumal and Radu Dragomir, 2024--2026.
% Contributors: Xiaowen Jiang and Bonan Sun
% Change log:
%
%   Sep. 11, 2024 (NB, RD):
%       Forked from trs_tCG.


% trustregions only wants header and default values for stats.
if nargin == 3 && isempty(problem) && isempty(trsinput)
    trsoutput.printheader = sprintf('%9s   %9s   %s', ...
                                    'numinner', 'hessvec', 'stopreason');
    trsoutput.initstats = struct('numinner', 0, 'hessvecevals', 0);
    return;
end

M = problem.M;

hessvecevals = 0;

x = trsinput.x;
Delta = trsinput.Delta;
grad = trsinput.fgradx;
norm_g = M.norm(x, grad);

inner   = @(u, v) M.inner(x, u, v);
lincomb = @(a, u, b, v) M.lincomb(x, a, u, b, v);
tangent = @(u) M.tangent(x, u);

% Set local defaults here
localdefaults.kappa = 0.1;
localdefaults.theta = 1.0;
localdefaults.mininner = 1;
localdefaults.maxinner = M.dim();
localdefaults.noiselevel = 100*sqrt(eps);
localdefaults.hessianshift = 0;

% Merge local defaults with user options, if any
if ~exist('options', 'var') || isempty(options)
    options = struct();
end
options = mergeOptions(localdefaults, options);

theta = options.theta;
kappa = options.kappa;

% If hessianshift is nonzero, then we add that multiple of the identity to
% the Hessian.
if options.hessianshift == 0
    hessian = @(u) getHessian(problem, x, u, storedb, key);
else
    hessian = @(u) problem.M.lincomb(x, ...
                            1, getHessian(problem, x, u, storedb, key), ...
                            options.hessianshift, u);
end

% This boolean is part of the outputs. It is set to true if the solution
% eta we return was (at least initially) limited in norm by the
% trust-region radius Delta/2.
limitedbyTR = false;

% Flip this to true if we encountered difficulties due to inexact
% arithmetic: trustregions may want to terminate if so.
numericaltrouble = false;

% Decide on the level of noise to add.
% Explicitly: we initialize tCG with a random vector of a certain norm,
% and right now we must decide what that norm shall be.
noiselevel = options.noiselevel;
if noiselevel < sqrt(eps)
    warning('trs_tCG_randomized:noiseleveleps', ...
      ['options.noiselevel seems small compared to machine precision.\n'...
       'Ideally, it should be larger than sqrt(eps).']);
end
if noiselevel > 1e-2*Delta
    warning('trs_tCG_randomized:noiselevel', ...
            'options.noiselevel seems large compared to Delta.');
end
noiselevel_old = noiselevel;
noiselevel = min(1e-2*Delta, max(sqrt(eps), noiselevel));
if noiselevel ~= noiselevel_old
    fprintf('noiselevel was %.3g; is now %.3g for this iteration.\n', ...
            noiselevel_old, noiselevel);
end


% Generate a random initial tangent vector of norm 1.
eta = M.randvec(x);
Heta = hessian(eta);
hessvecevals = hessvecevals + 1;

% Scale eta and Heta to the desired noise level.
% Possibly also flip their sign to make sure that the sum of grad and Heta
% is at least as large as grad itself.
if inner(Heta, grad) >= 0
    eta = M.lincomb(x, noiselevel, eta);
    Heta = M.lincomb(x, noiselevel, Heta);
else
    eta = M.lincomb(x, -noiselevel, eta);
    Heta = M.lincomb(x, -noiselevel, Heta);
end

% Squared norm of eta.
e_e = noiselevel^2;

r = lincomb(1, grad, 1, Heta);
r_r = inner(r, r);

% Initial search direction (we maintain -delta in memory, called mdelta, to
% avoid a change of sign of the tangent vector.)
mdelta = r;
d_d = r_r;
e_d = -inner(eta, mdelta);

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
model_value = model_fun(eta, Heta);

% !!
% In this randomized version of tCG, the initial eta is nonzero, and
% therefore the initial model value is nonzero. We return that value to
% trustregions to be used as a regularizer in the computation of rho.
trsoutput.rho_reg = model_value;

% Pre-assume termination because j == end.
stopreason_str = '<strong>maximum inner iterations</strong>';

% If the initial residual is zero (highly unlikely), we terminate now.
if r_r == 0
    stopreason_str = 'initial residual is zero.';
else

% Begin inner/tCG loop.
for j = 1 : options.maxinner
    
    % This call is the computationally expensive step.
    Hmdelta = hessian(mdelta);
    hessvecevals = hessvecevals + 1;
    
    % Compute curvature.
    d_Hd = inner(mdelta, Hmdelta);
    
    
    % Note that if d_Hd == 0, we will exit at the next "if" anyway.
    alpha = r_r/d_Hd;
    e_e_new = e_e + 2*alpha*e_d + alpha*alpha*d_d;
    
    % Check against negative curvature and trust-region radius violation.
    % If either condition triggers, we bail out, but in a new way.
    % Notice that our initial trust region redius is Delta/2, not Delta.
    if d_Hd <= 0 || e_e_new >= (Delta/2)^2

        % We tell trustregions that we were limited by the TR radius.
        % This is indeed the case for the radius Delta/2.
        % Possibly, the extra Cauchy step performed next might not be
        % limited by the larger ball (of radius Delta), but that is
        % irrelevant for this particular determination.
        limitedbyTR = true;

        % Move up to (or bactrack to) radius Delta/2.
        tau = max(real(roots([d_d, 2*e_d, e_e-(Delta/2)^2])));
        eta = lincomb(1, eta, -tau, mdelta);
        
        % If only a nonlinear Hessian approximation is available, this is
        % only approximately correct, but saves an additional Hessian call.
        Heta = lincomb(1, Heta, -tau, Hmdelta);

        % This base message reports on the first stop reason (at Delta/2).
        if d_Hd <= 0
            stopreason_str = 'negative curvature (Delta/2)';
        else
            stopreason_str = 'exceeded TR (Delta/2)';
        end

        % Now we perform an extra gradient/Cauchy step in a Delta ball.
        % The starting point is eta: it is on the boundary of the ball of
        % radius Delta/2 (by construction). Compute q: the gradient of the
        % model at eta.
        q = lincomb(1, Heta, 1, grad);
        Hq = hessian(q);
        hessvecevals = hessvecevals + 1;
        qHq = inner(q, Hq);
        q_q = inner(q, q);
        qeta = inner(q, eta);
        tau = q_q / qHq;
        % In principle, we move to eta + tau*q, unless the curvature of the
        % model along q is nonpositive or if that would bring us outside
        % the ball of radius Delta. If either of those triggers, we move to
        % the boundary of the ball of radius Delta instead.
        sqn = (Delta/2)^2 + tau^2*q_q - 2*tau*qeta;
        if qHq <= 0 || sqn >= Delta^2
            tau = max(real(roots([q_q, -2*qeta, (Delta/2)^2 - Delta^2])));
        end
        eta = lincomb(1, eta, -tau, q);
        Heta = lincomb(1, Heta, -tau, Hq);

        % This extra message reports on the extra Cauchy step (in Delta).
        if qHq <= 0
            stopreason_str = [stopreason_str, ', ', ...
                              'negative curvature (Delta)']; %#ok<AGROW>
        elseif sqn >= Delta^2
            stopreason_str = [stopreason_str, ', ', ...
                              'exceeded TR (Delta)']; %#ok<AGROW>
        else
            stopreason_str = [stopreason_str, ', ', ...
                              'extra Cauchy in TR (Delta)']; %#ok<AGROW>
        end

        break;
    end
    
    % No negative curvature and eta_prop inside TR: accept it.
    new_eta = lincomb(1, eta, -alpha,  mdelta);
    
    % If only a nonlinear Hessian approximation is available, this is
    % only approximately correct, but saves an additional Hessian call.
    new_Heta = lincomb(1, Heta, -alpha, Hmdelta);
    
    % Verify that the model cost decreased in going from eta to new_eta.
    % If it did not (which can only occur if the Hessian approximation is
    % nonlinear or because of numerical errors), then we return the
    % previous eta (which necessarily is the best reached so far, according
    % to the model cost). Otherwise, we accept the new eta and go on.
    new_model_value = model_fun(new_eta, new_Heta);
    if new_model_value > model_value
        stopreason_str = 'model increased in last inner iteration';
        numericaltrouble = true;
        break;
    end
    
    eta = new_eta;
    e_e = e_e_new;
    Heta = new_Heta;
    model_value = new_model_value;
    
    % Update the residual.
    r = lincomb(1, r, -alpha, Hmdelta);
    
    % Save the old r'*r
    rold_rold = r_r;
    % Compute new norm of r.
    r_r = inner(r, r);
    norm_r = sqrt(r_r);
    
    % Check kappa/theta stopping criterion.
    if j >= options.mininner && norm_r <= norm_g*min(norm_g^theta, kappa)
        % Residual is small enough to quit
        if kappa < norm_g^theta
            stopreason_str = 'reached target residual-kappa (linear)';
        else
            stopreason_str = 'reached target residual-theta (superlinear)';
        end
        break;
    end
    
    % Compute new search direction.
    beta = r_r/rold_rold;
    mdelta = lincomb(1, r, beta, mdelta);
    
    % Since mdelta is passed to getHessian, which is the part of the code
    % we have least control over from here, we want to make sure mdelta is
    % a tangent vector up to numerical errors that should remain small.
    % For this reason, we re-project mdelta to the tangent space.
    % In limited tests, it was observed that it is a good idea to project
    % at every iteration rather than only every k iterations, the reason
    % being that loss of tangency can lead to more inner iterations being
    % run, which leads to an overall higher computational cost.
    mdelta = tangent(mdelta);
    
    e_d = -inner(eta, mdelta);
    d_d = r_r + beta*beta*d_d;
    
end  % of tCG loop

end % if/else on r_r == 0


printstr = sprintf('%9d   %9d   %s', j, j, stopreason_str);
stats = struct('numinner', j, 'hessvecevals', hessvecevals);

trsoutput.eta = eta;
trsoutput.Heta = Heta;
trsoutput.limitedbyTR = limitedbyTR;
trsoutput.numericaltrouble = numericaltrouble;
trsoutput.printstr = printstr;
trsoutput.stats = stats;

end
