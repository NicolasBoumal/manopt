function trsoutput = trs_lanczos(problem, trsinput, options, storedb, key)
% Generalized Lanczos trust-region method GLTR for trustregions subproblem.
%
% minimize <eta,grad> + .5*<eta,Hess(eta)>
% subject to <eta,eta>_[inverse precon] <= Delta^2
%
% function trsoutput = trs_lanczos(problem, trsinput, options, storedb, key)
%
% Inputs:
%   problem: Manopt optimization problem structure
%   trsinput: structure with the following fields:
%       x: point on the manifold problem.M
%       fgradx: gradient of the cost function of the problem at x
%       Delta = trust-region radius
%   options: structure containing options for the subproblem solver
%   storedb, key: manopt's caching system for the point x
%
% Options specific to this subproblem solver:
%   kappa (0.1)
%       kappa convergence tolerance.
%       kappa > 0 is the linear convergence target rate: lanczos
%       terminates early if the residual was reduced by a factor of kappa.
%   theta (1.0)
%       theta convergence tolerance.
%       1+theta (theta between 0 and 1) is the superlinear convergence
%       target rate. lanczos terminates early if the residual 
%       was reduced by a power of 1+theta.
%   mininner (1)
%       Minimum number of inner iterations.
%   maxinner (problem.M.dim())
%       Maximum number of inner iterations.
%   maxiter_newton (100)
%     Maximum number of iterations of the Newton root finder to solve each
%     tridiagonal quadratic problem.
%   tol_newton (1e-16)
%     Tolerance for the Newton root finder.
%
% Output: the structure trsoutput contains the following fields:
%   eta: approximate solution to the trust-region subproblem at x
%   Heta: Hess f(x)[eta] -- this is necessary in the outer loop, and it
%       is often naturally available to the subproblem solver at the
%       end of execution, so that it may be cheaper to return it here.
%   limitedbyTR: true if a boundary solution is returned
%   printstr: logged information to be printed by trustregions.
%   stats: structure with the following statistics:
%           numinner: number of inner loops before returning
%           hessvecevals: number of Hessian calls issued
%
%
% trs_lanczos can also be called in the following way (by trustregions) 
% to obtain part of the header to print and an initial stats structure:
%
% function trsoutput = trs_lanczos([], [], options)
%
% In this case trsoutput contains the following fields:
%   printheader: subproblem header to be printed before the first pass of 
%       trustregions
%   initstats: struct with initial values for stored stats in subsequent
%       calls to trs_lanczos. Used in the first call to savestats 
%       in trustregions to initialize the info struct properly.
%
% See also: trustregions minimize_quadratic_newton trs_tCG

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

% See trs_tCG for references to some relevant equations in
% [CGT2000] Conn, Gould and Toint: Trust-region methods, 2000.
%
% Original paper describing the algorithm and equations on converting from 
% tCG to lanczos:
% [GLRT1999] Gould, Lucidi, Roma, Toint: SOLVING THE TRUST-REGION 
% SUBPROBLEM USING THE LANCZOS METHOD, 1999.

if nargin == 3 && isempty(problem) && isempty(trsinput)
    trsoutput.printheader = sprintf('%9s   %9s   %s', 'numinner', ...
                            'hessvec', 'stopreason');
    trsoutput.initstats = struct('numinner', 0, 'hessvecevals', 0);
    return;
end

x = trsinput.x;
Delta = trsinput.Delta;
grad = trsinput.fgradx;

M = problem.M;
n = M.dim();

inner   = @(u, v) M.inner(x, u, v);
tangent = @(u) M.tangent(x, u);

% Set local defaults here
localdefaults.kappa = 0.1;
localdefaults.theta = 1.0;
localdefaults.mininner = 1;
localdefaults.maxinner = M.dim();
localdefaults.maxiter_newton = 100;
localdefaults.tol_newton = 1e-16;

% Merge local defaults with user options, if any
if ~exist('options', 'var') || isempty(options)
    options = struct();
end
options = mergeOptions(localdefaults, options);

theta = options.theta;
kappa = options.kappa;

eta = M.zerovec(x);
Heta = M.zerovec(x);
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

% gamma_0 in lanczos tridiagonal problem
gamma_0 = sqrt(z_r);

% Initial search direction (we maintain -delta in memory, called mdelta, to
% avoid a change of sign of the tangent vector.)
mdelta = z;
e_Pd = 0;

% interior being false means we solve tridiagonal trust-region subproblem
% and generate h
interior = true;

% Lanczos iteratively produces an orthonormal basis of tangent vectors
% which tridiagonalize the Hessian. The corresponding tridiagonal
% matrix is preallocated here as a sparse matrix.
T = spdiags(zeros(n, 3), -1:1, n, n);

% store Lanczos vectors
Q = cell(n, 1);

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
model_fun_h = @(eta, g, H) dot(eta, g) + .5* dot(eta, H * eta);
model_value = 0;

% Pre-assume termination because j == end.
stopreason_str = 'maximum inner iterations';

% This call is the computationally expensive step.
Hmdelta = getHessian(problem, x, mdelta, storedb, key);

% Compute curvature (often called kappa).
d_Hd = inner(mdelta, Hmdelta);

alpha = z_r/d_Hd;

% Begin inner loop.
for j = 1 : min(options.maxinner, n)

    % obtain T_k from T_{k-1}
    if j == 1
        T(j, j) = 1/alpha;
        Q{j} = M.lincomb(x, 1/sqrt(z_r), z);
        sigma_k = -sign(alpha);
    else
        T(j-1, j) = sqrt(beta)/abs(prevalpha); %sqrt(beta_{j-1})/abs(alpha_{j-1})
        T(j, j-1) = sqrt(beta)/abs(prevalpha); %sqrt(beta_{j-1})/abs(alpha_{j-1})
        T(j, j) = 1/alpha + beta/prevalpha;
        
        q = M.lincomb(x, sigma_k/sqrt(z_r), z);
        sigma_k = - sign(alpha) * sigma_k;
        q = tangent(q);
        
        Q{j} = q;
    end
    
    if options.debug > 2
        fprintf('DBG:   (r,r)  : %e\n', r_r);
        fprintf('DBG:   (d,Hd) : %e\n', d_Hd);
        fprintf('DBG:   alpha  : %e\n', alpha);
    end

    if interior
        % <neweta,neweta>_P =
        % <eta,eta>_P + 2*alpha*<eta,delta>_P + alpha*alpha*<delta,delta>_P
        e_Pe_new = e_Pe + 2.0*alpha*e_Pd + alpha*alpha*d_Pd;
        
        % Check against negative curvature and trust-region radius violation.
        % If either condition triggers, we switch to lanczos.
        if (alpha <= 0 || e_Pe_new >= Delta^2)
            interior = false;
        else
            % No negative curvature and eta_prop inside TR: accept it.
            e_Pe = e_Pe_new;
            new_eta  = M.lincomb(x, 1, eta, -alpha, mdelta);
        
            % If only a nonlinear Hessian approximation is available, this is
            % only approximately correct, but saves an additional Hessian call.
            % TODO: this computation is redundant with that of r, L241. Clean up.
            new_Heta = M.lincomb(x, 1, Heta, -alpha, Hmdelta);
        
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
        end
    end
    
    % solve tridiagonal trust-region subproblem to obtain h
    if ~interior
        [new_h, ~] = minimize_quadratic_newton(T(1:j, 1:j), ...
                               gamma_0*eye(j, 1), Delta, options);
        
        new_model_value = model_fun_h(new_h, gamma_0*eye(j, 1), T(1:j, 1:j));
        if new_model_value >= model_value
            stopreason_str = 'model increased';
            break;
        end
        h = new_h;
        model_value = new_model_value;
    end

    % Update the residual.
    r = M.lincomb(x, 1, r, -alpha, Hmdelta);
    
    % Compute new norm of r.
    r_r = inner(r, r);
    norm_r = sqrt(r_r);

    % Precondition the residual.
    z = getPrecon(problem, x, r, storedb, key);
    
    % Save the old z'*r.
    zold_rold = z_r;

    % Compute new z'*r.
    z_r = inner(z, r);

    beta = z_r/zold_rold;

    if interior
        % Check kappa/theta stopping criterion.
        % Note that it is somewhat arbitrary whether to check this stopping
        % criterion on the r's (the gradients) or on the z's (the
        % preconditioned gradients). [CGT2000], page 206, mentions both as
        % acceptable criteria.
        conv_test = norm_r;
    else
        % gamma_{j} |< e_{j}, h_{j-1}|
        conv_test = sqrt(beta)/abs(alpha) * abs(dot(double(1:j == j), h));
    end

    if j >= options.mininner && conv_test <= norm_r0*min(norm_r0^theta, kappa)
        % Residual is small enough to quit
        if kappa < norm_r0^theta
            stopreason_str = 'reached target residual-kappa (linear)';
        else
            stopreason_str = 'reached target residual-theta (superlinear)';
        end
        break;
    end

    % Compute new search direction.
    mdelta = M.lincomb(x, 1, z, beta, mdelta);
    
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
    
    % This call is the computationally expensive step.
    Hmdelta = getHessian(problem, x, mdelta, storedb, key);

    % Compute curvature (often called kappa).
    d_Hd = inner(mdelta, Hmdelta);

    prevalpha = alpha;
    alpha = z_r/d_Hd;

    % Pre-assume termination because j == end.
    stopreason_str = 'maximum inner iterations';
end  % of loop

% recover the solution eta = Q_k h_k
if ~interior
    eta = lincomb(M, x, Q(1:numel(h)), h);
    eta = tangent(eta);
    Heta = getHessian(problem, x, eta, storedb, key);
    stopreason_str = append(stopreason_str,' (lanczos)');
else
    stopreason_str = append(stopreason_str,' (tCG)');
end

printstr = sprintf('%9d   %9d   %s', j, j, stopreason_str);
stats = struct('numinner', j, 'hessvecevals', j);

trsoutput.eta = eta;
trsoutput.Heta = Heta;
trsoutput.limitedbyTR = ~interior;
trsoutput.printstr = printstr;
trsoutput.stats = stats;
end
