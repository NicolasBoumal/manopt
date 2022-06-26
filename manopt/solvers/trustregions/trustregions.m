function [x, cost, info, options] = trustregions(problem, x, options)
% Riemannian trust-regions solver for optimization on manifolds.
%
% function [x, cost, info, options] = trustregions(problem)
% function [x, cost, info, options] = trustregions(problem, x0)
% function [x, cost, info, options] = trustregions(problem, x0, options)
% function [x, cost, info, options] = trustregions(problem, [], options)
%
% This is the Riemannian Trust-Region solver (with tCG inner solve), named
% RTR. This solver will attempt to minimize the cost function described in
% the problem structure. It requires the availability of the cost function
% and of its gradient. It will issue calls for the Hessian. If no Hessian
% nor approximate Hessian is provided, a standard approximation of the
% Hessian based on the gradient will be computed. If a preconditioner for
% the Hessian is provided, it will be used.
%
% If no gradient is provided, an approximation of the gradient is computed,
% but this can be slow for manifolds of high dimension.
%
% For a description of the algorithm and theorems offering convergence
% guarantees, see the references below. Documentation for this solver is
% available online at:
%
% http://www.manopt.org/solver_documentation_trustregions.html
%
%
% The initial iterate is x0 if it is provided. Otherwise, a random point on
% the manifold is picked. To specify options whilst not specifying an
% initial iterate, give x0 as [] (the empty matrix).
%
% The two outputs 'x' and 'cost' are the last reached point on the manifold
% and its cost. Notice that x is not necessarily the best reached point,
% because this solver is not forced to be a descent method. In particular,
% very close to convergence, it is sometimes preferable to accept very
% slight increases in the cost value (on the order of the machine epsilon)
% in the process of reaching fine convergence.
% 
% The output 'info' is a struct-array which contains information about the
% iterations:
%   iter (integer)
%       The (outer) iteration number, or number of steps considered
%       (whether accepted or rejected). The initial guess is 0.
%   cost (double)
%       The corresponding cost value.
%   gradnorm (double)
%       The (Riemannian) norm of the gradient.
%   numinner (integer)
%       The number of inner iterations executed to compute this iterate.
%       Inner iterations are truncated-CG steps. Each one requires a
%       Hessian (or approximate Hessian) evaluation.
%   time (double)
%       The total elapsed time in seconds to reach the corresponding cost.
%   rho (double)
%       The performance ratio for the iterate.
%   rhonum, rhoden (double)
%       Regularized numerator and denominator of the performance ratio:
%       rho = rhonum/rhoden. See options.rho_regularization.
%   accepted (boolean)
%       Whether the proposed iterate was accepted or not.
%   stepsize (double)
%       The (Riemannian) norm of the vector returned by the inner solver
%       tCG and which is retracted to obtain the proposed next iterate. If
%       accepted = true for the corresponding iterate, this is the size of
%       the step from the previous to the new iterate. If accepted is
%       false, the step was not executed and this is the size of the
%       rejected step.
%   Delta (double)
%       The trust-region radius at the outer iteration.
%   cauchy (boolean)
%       Whether the Cauchy point was used or not (if useRand is true).
%   And possibly additional information logged by options.statsfun.
% For example, type [info.gradnorm] to obtain a vector of the successive
% gradient norms reached at each (outer) iteration.
%
% The options structure is used to overwrite the default values. All
% options have a default value and are hence optional. To force an option
% value, pass an options structure with a field options.optionname, where
% optionname is one of the following and the default value is indicated
% between parentheses:
%
%   tolgradnorm (1e-6)
%       The algorithm terminates if the norm of the gradient drops below
%       this. For well-scaled problems, a rule of thumb is that you can
%       expect to reduce the gradient norm by 8 orders of magnitude
%       (sqrt(eps)) compared to the gradient norm at a "typical" point (a
%       rough initial iterate for example). Further decrease is sometimes
%       possible, but inexact floating point arithmetic will eventually
%       limit the final accuracy. If tolgradnorm is set too low, the
%       algorithm may end up iterating forever (or at least until another
%       stopping criterion triggers).
%   maxiter (1000)
%       The algorithm terminates if maxiter (outer) iterations were executed.
%   maxtime (Inf)
%       The algorithm terminates if maxtime seconds elapsed.
%   miniter (3)
%       Minimum number of outer iterations (used only if useRand is true).
%   mininner (1)
%       Minimum number of inner iterations (for tCG).
%   maxinner (problem.M.dim() : the manifold's dimension)
%       Maximum number of inner iterations (for tCG).
%   Delta_bar (problem.M.typicaldist() or sqrt(problem.M.dim()))
%       Maximum trust-region radius. If you specify this parameter but not
%       Delta0, then Delta0 will be set to 1/8 times this parameter.
%   Delta0 (Delta_bar/8)
%       Initial trust-region radius. If you observe a long plateau at the
%       beginning of the convergence plot (gradient norm VS iteration), it
%       may pay off to try to tune this parameter to shorten the plateau.
%       You should not set this parameter without setting Delta_bar too (at
%       a larger value).
%   useRand (false)
%       Set to true if the trust-region solve is to be initiated with a
%       random tangent vector. If set to true, no preconditioner will be
%       used. This option is set to true in some scenarios to escape saddle
%       points, but is otherwise seldom activated.
%   kappa (0.1)
%       tCG inner kappa convergence tolerance.
%       kappa > 0 is the linear convergence target rate: tCG will terminate
%       early if the residual was reduced by a factor of kappa.
%   theta (1.0)
%       tCG inner theta convergence tolerance.
%       1+theta (theta between 0 and 1) is the superlinear convergence
%       target rate. tCG will terminate early if the residual was reduced
%       by a power of 1+theta.
%   rho_prime (0.1)
%       Accept/reject threshold : if rho is at least rho_prime, the outer
%       iteration is accepted. Otherwise, it is rejected. In case it is
%       rejected, the trust-region radius will have been decreased.
%       To ensure this, rho_prime >= 0 must be strictly smaller than 1/4.
%       If rho_prime is negative, the algorithm is not guaranteed to
%       produce monotonically decreasing cost values. It is strongly
%       recommended to set rho_prime > 0, to aid convergence.
%   rho_regularization (1e3)
%       Close to convergence, evaluating the performance ratio rho is
%       numerically challenging. Meanwhile, close to convergence, the
%       quadratic model should be a good fit and the steps should be
%       accepted. Regularization lets rho go to 1 as the model decrease and
%       the actual decrease go to zero. Set this option to zero to disable
%       regularization (not recommended). See in-code for the specifics.
%       When this is not zero, it may happen that the iterates produced are
%       not monotonically improving the cost when very close to
%       convergence. This is because the corrected cost improvement could
%       change sign if it is negative but very small.
%   statsfun (none)
%       Function handle to a function that will be called after each
%       iteration to provide the opportunity to log additional statistics.
%       They will be returned in the info struct. See the generic Manopt
%       documentation about solvers for further information. statsfun is
%       called with the point x that was reached last, after the
%       accept/reject decision. See comment below.
%   stopfun (none)
%       Function handle to a function that will be called at each iteration
%       to provide the opportunity to specify additional stopping criteria.
%       See the generic Manopt documentation about solvers for further
%       information.
%   verbosity (2)
%       Integer number used to tune the amount of output the algorithm
%       generates during execution (mostly as text in the command window).
%       The higher, the more output. 0 means silent. 3 and above includes a
%       display of the options structure at the beginning of the execution.
%   debug (false)
%       Set to true to allow the algorithm to perform additional
%       computations for debugging purposes. If a debugging test fails, you
%       will be informed of it, usually via the command window. Be aware
%       that these additional computations appear in the algorithm timings
%       too, and may interfere with operations such as counting the number
%       of cost evaluations, etc. (the debug calls get storedb too).
%   storedepth (2)
%       Maximum number of different points x of the manifold for which a
%       store structure will be kept in memory in the storedb for caching.
%       If memory usage is an issue, you may try to lower this number.
%       Profiling or manopt counters may then help to investigate if a
%       performance hit was incurred as a result.
%   hook (none)
%       A function handle which allows the user to change the current point
%       x at the beginning of each iteration, before the stopping criterion
%       is evaluated. See applyHook for help on how to use this option.
%
% Notice that statsfun is called with the point x that was reached last,
% after the accept/reject decision. Hence: if the step was accepted, we get
% that new x, with a store which only saw the call for the cost and for the
% gradient. If the step was rejected, we get the same x as previously, with
% the store structure containing everything that was computed at that point
% (possibly including previous rejects at that same point). Hence, statsfun
% should not be used in conjunction with the store to count operations for
% example. Instead, you should use manopt counters: see statscounters.
%
%
% Please cite the Manopt paper as well as the research paper:
%     @Article{genrtr,
%       Title    = {Trust-region methods on {Riemannian} manifolds},
%       Author   = {Absil, P.-A. and Baker, C. G. and Gallivan, K. A.},
%       Journal  = {Foundations of Computational Mathematics},
%       Year     = {2007},
%       Number   = {3},
%       Pages    = {303--330},
%       Volume   = {7},
%       Doi      = {10.1007/s10208-005-0179-9}
%     }
%
% See also: steepestdescent conjugategradient manopt/examples

% An explicit, general listing of this algorithm, with preconditioning,
% can be found in the following paper:
%     @Article{boumal2015lowrank,
%       Title   = {Low-rank matrix completion via preconditioned optimization on the {G}rassmann manifold},
%       Author  = {Boumal, N. and Absil, P.-A.},
%       Journal = {Linear Algebra and its Applications},
%       Year    = {2015},
%       Pages   = {200--239},
%       Volume  = {475},
%       Doi     = {10.1016/j.laa.2015.02.027},
%     }

% When the Hessian is not specified, it is approximated with
% finite-differences of the gradient. The resulting method is called
% RTR-FD. Some convergence theory for it is available in this paper:
% @incollection{boumal2015rtrfd
%    author={Boumal, N.},
%    title={Riemannian trust regions with finite-difference Hessian approximations are globally convergent},
%    year={2015},
%    booktitle={Geometric Science of Information}
% }


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
%
% Change log: 
%
%   NB April 3, 2013:
%       tCG now returns the Hessian along the returned direction eta, so
%       that we do not compute that Hessian redundantly: some savings at
%       each iteration. Similarly, if the useRand flag is on, we spare an
%       extra Hessian computation at each outer iteration too, owing to
%       some modifications in the Cauchy point section of the code specific
%       to useRand = true.
%
%   NB Aug. 22, 2013:
%       This function is now Octave compatible. The transition called for
%       two changes which would otherwise not be advisable. (1) tic/toc is
%       now used as is, as opposed to the safer way:
%       t = tic(); elapsed = toc(t);
%       And (2), the (formerly inner) function savestats was moved outside
%       the main function to not be nested anymore. This is arguably less
%       elegant, but Octave does not (and likely will not) support nested
%       functions.
%
%   NB Dec. 2, 2013:
%       The in-code documentation was largely revised and expanded.
%
%   NB Dec. 2, 2013:
%       The former heuristic which triggered when rhonum was very small and
%       forced rho = 1 has been replaced by a smoother heuristic which
%       consists in regularizing rhonum and rhoden before computing their
%       ratio. It is tunable via options.rho_regularization. Furthermore,
%       the solver now detects if tCG did not obtain a model decrease
%       (which is theoretically impossible but may happen because of
%       numerical errors and/or because of a nonlinear/nonsymmetric Hessian
%       operator, which is the case for finite difference approximations).
%       When such an anomaly is detected, the step is rejected and the
%       trust region radius is decreased.
%       Feb. 18, 2015 note: this is less useful now, as tCG now guarantees
%       model decrease even for the finite difference approximation of the
%       Hessian. It is still useful in case of numerical errors, but this
%       is less stringent.
%
%   NB Dec. 3, 2013:
%       The stepsize is now registered at each iteration, at a small
%       additional cost. The defaults for Delta_bar and Delta0 are better
%       defined. Setting Delta_bar in the options will automatically set
%       Delta0 accordingly. In Manopt 1.0.4, the defaults for these options
%       were not treated appropriately because of an incorrect use of the
%       isfield() built-in function.
%
%   NB Feb. 18, 2015:
%       Added some comments. Also, Octave now supports safe tic/toc usage,
%       so we reverted the changes to use that again (see Aug. 22, 2013 log
%       entry).
%
%   NB April 3, 2015:
%       Works with the new StoreDB class system.
%
%   NB April 8, 2015:
%       No Hessian warning if approximate Hessian explicitly available.
%
%   NB Nov. 1, 2016:
%       Now uses approximate gradient via finite differences if need be.
%
%   NB Aug. 2, 2018:
%       Using storedb.remove() to keep the cache lean, which allowed to
%       reduce storedepth to 2 from 20 (by default).
%
%   NB July 19, 2020:
%       Added support for options.hook.

M = problem.M;

% Verify that the problem description is sufficient for the solver.
if ~canGetCost(problem)
    warning('manopt:getCost', ...
            'No cost provided. The algorithm will likely abort.');  
end
if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
    % Note: we do not give a warning if an approximate gradient is
    % explicitly given in the problem description, as in that case the user
    % seems to be aware of the issue.
    warning('manopt:getGradient:approx', ...
           ['No gradient provided. Using an FD approximation instead (slow).\n' ...
            'It may be necessary to increase options.tolgradnorm.\n' ...
            'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
    problem.approxgrad = approxgradientFD(problem);
end
if ~canGetHessian(problem) && ~canGetApproxHessian(problem)
    % Note: we do not give a warning if an approximate Hessian is
    % explicitly given in the problem description, as in that case the user
    % seems to be aware of the issue.
    warning('manopt:getHessian:approx', ...
           ['No Hessian provided. Using an FD approximation instead.\n' ...
            'To disable this warning: warning(''off'', ''manopt:getHessian:approx'')']);
    problem.approxhess = approxhessianFD(problem);
end

% Define some strings for display
tcg_stop_reason = {'negative curvature',...
                   'exceeded trust region',...
                   'reached target residual-kappa (linear)',...
                   'reached target residual-theta (superlinear)',...
                   'maximum inner iterations',...
                   'model increased'};

% Set local defaults here
localdefaults.verbosity = 2;
localdefaults.maxtime = inf;
localdefaults.miniter = 3;
localdefaults.maxiter = 1000;
localdefaults.mininner = 1;
localdefaults.maxinner = M.dim();
localdefaults.tolgradnorm = 1e-6;
localdefaults.kappa = 0.1;
localdefaults.theta = 1.0;
localdefaults.rho_prime = 0.1;
localdefaults.useRand = false;
localdefaults.rho_regularization = 1e3;
localdefaults.accept = true;

% Merge global and local defaults, then merge w/ user options, if any.
localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
if ~exist('options', 'var') || isempty(options)
    options = struct();
end
options = mergeOptions(localdefaults, options);

% Set default Delta_bar and Delta0 separately to deal with additional
% logic: if Delta_bar is provided but not Delta0, let Delta0 automatically
% be some fraction of the provided Delta_bar.
if ~isfield(options, 'Delta_bar')
    if isfield(M, 'typicaldist')
        options.Delta_bar = M.typicaldist();
    else
        options.Delta_bar = sqrt(M.dim());
    end 
end
if ~isfield(options,'Delta0')
    options.Delta0 = options.Delta_bar / 8;
end

% Check some option values
assert(options.rho_prime < 1/4, ...
        'options.rho_prime must be strictly smaller than 1/4.');
assert(options.Delta_bar > 0, ...
        'options.Delta_bar must be positive.');
assert(options.Delta0 > 0 && options.Delta0 <= options.Delta_bar, ...
        'options.Delta0 must be positive and smaller than Delta_bar.');

% It is sometimes useful to check what the actual option values are.
if options.verbosity >= 3
    disp(options);
end

ticstart = tic();

% If no initial point x is given by the user, generate one at random.
if ~exist('x', 'var') || isempty(x)
    x = M.rand();
end

% Create a store database and get a key for the current x
storedb = StoreDB(options.storedepth);
key = storedb.getNewKey();

%% Initializations

% k counts the outer (TR) iterations. The semantic is that k counts the
% number of iterations fully executed so far.
k = 0;

% Initialize solution and companion measures: f(x), fgrad(x)
[fx, fgradx] = getCostGrad(problem, x, storedb, key);
norm_grad = M.norm(x, fgradx);

% Initialize trust-region radius
Delta = options.Delta0;

% Save stats in a struct array info, and preallocate.
if ~exist('used_cauchy', 'var')
    used_cauchy = [];
end
stats = savestats(problem, x, storedb, key, options, k, fx, norm_grad, Delta, ticstart);
info(1) = stats;
info(min(10000, options.maxiter+1)).iter = [];

% ** Display:
if options.verbosity == 2
   fprintf(['%3s %3s      %5s                %5s     ',...
            'f: %+e   |grad|: %e\n'],...
           '   ','   ','     ','     ', fx, norm_grad);
elseif options.verbosity > 2
   fprintf('************************************************************************\n');
   fprintf('%3s %3s    k: %5s     num_inner: %5s     %s\n',...
           '','','______','______','');
   fprintf('       f(x) : %+e       |grad| : %e\n', fx, norm_grad);
   fprintf('      Delta : %f\n', Delta);
end

% To keep track of consecutive radius changes, so that we can warn the
% user if it appears necessary.
consecutive_TRplus = 0;
consecutive_TRminus = 0;


% **********************
% ** Start of TR loop **
% **********************
while true
    
    % Start clock for this outer iteration
    ticstart = tic();

    % Apply the hook function if there is one: this allows external code to
    % move x to another point. If the point is changed (indicated by a true
    % value for the boolean 'hooked'), we update our knowledge about x.
    [x, key, info, hooked] = applyHook(problem, x, storedb, key, options, info, k+1);
    if hooked
        [fx, fgradx] = getCostGrad(problem, x, storedb, key);
        norm_grad = M.norm(x, fgradx);
    end
    
    % Run standard stopping criterion checks
    [stop, reason] = stoppingcriterion(problem, x, options, info, k+1);
    
    % If the stopping criterion that triggered is the tolerance on the
    % gradient norm but we are using randomization, make sure we make at
    % least miniter iterations to give randomization a chance at escaping
    % saddle points.
    if stop == 2 && options.useRand && k < options.miniter
        stop = 0;
    end
    
    if stop
        if options.verbosity >= 1
            fprintf([reason '\n']);
        end
        break;
    end

    if options.verbosity > 2 || options.debug > 0
        fprintf('************************************************************************\n');
    end

    % *************************
    % ** Begin TR Subproblem **
    % *************************
  
    % Determine eta0
    if ~options.useRand
        % Pick the zero vector
        eta = M.zerovec(x);
    else
        % Random vector in T_x M (this has to be very small)
        eta = M.lincomb(x, 1e-6, M.randvec(x));
        % Must be inside trust-region
        while M.norm(x, eta) > Delta
            eta = M.lincomb(x, sqrt(sqrt(eps)), eta);
        end
    end
    % Solve TR subproblem approximately

    [eta, Heta, numit, stop_inner] = ...
                tCG(problem, x, fgradx, eta, Delta, options, storedb, key);
    srstr = tcg_stop_reason{stop_inner};

    % If using randomized approach, compare result with the Cauchy point.
    % Convergence proofs assume that we achieve at least (a fraction of)
    % the reduction of the Cauchy point. After this if-block, either all
    % eta-related quantities have been changed consistently, or none of
    % them have changed.
    if options.useRand
        used_cauchy = false;
        % Check the curvature,
        Hg = getHessian(problem, x, fgradx, storedb, key);
        g_Hg = M.inner(x, fgradx, Hg);
        if g_Hg <= 0
            tau_c = 1;
        else
            tau_c = min( norm_grad^3/(Delta*g_Hg) , 1);
        end
        % and generate the Cauchy point.
        eta_c  = M.lincomb(x, -tau_c * Delta / norm_grad, fgradx);
        Heta_c = M.lincomb(x, -tau_c * Delta / norm_grad, Hg);

        % Now that we have computed the Cauchy point in addition to the
        % returned eta, we might as well keep the best of them.
        mdle  = fx + M.inner(x, fgradx, eta) + .5*M.inner(x, Heta,   eta);
        mdlec = fx + M.inner(x, fgradx, eta_c) + .5*M.inner(x, Heta_c, eta_c);
        if mdlec < mdle
            eta = eta_c;
            Heta = Heta_c; % added April 11, 2012
            used_cauchy = true;
        end
    end
    
    
    % This is computed for logging purposes and may be useful for some
    % user-defined stopping criteria.
    norm_eta = M.norm(x, eta);
    
    if options.debug > 0
        testangle = M.inner(x, eta, fgradx) / (norm_eta*norm_grad);
    end
    

    % Compute the tentative next iterate (the proposal)
    x_prop  = M.retr(x, eta);
    key_prop = storedb.getNewKey();

    % Compute the function value of the proposal
    fx_prop = getCost(problem, x_prop, storedb, key_prop);

    % Will we accept the proposal or not?
    % Check the performance of the quadratic model against the actual cost.
    rhonum = fx - fx_prop;
    vecrho = M.lincomb(x, 1, fgradx, .5, Heta);
    rhoden = -M.inner(x, eta, vecrho);
    % rhonum could be anything.
    % rhoden should be nonnegative, as guaranteed by tCG, baring numerical
    % errors.
    
    % Heuristic -- added Dec. 2, 2013 (NB) to replace the former heuristic.
    % This heuristic is documented in the book by Conn Gould and Toint on
    % trust-region methods, section 17.4.2.
    % rhonum measures the difference between two numbers. Close to
    % convergence, these two numbers are very close to each other, so
    % that computing their difference is numerically challenging: there may
    % be a significant loss in accuracy. Since the acceptance or rejection
    % of the step is conditioned on the ratio between rhonum and rhoden,
    % large errors in rhonum result in a very large error in rho, hence in
    % erratic acceptance / rejection. Meanwhile, close to convergence,
    % steps are usually trustworthy and we should transition to a Newton-
    % like method, with rho=1 consistently. The heuristic thus shifts both
    % rhonum and rhoden by a small amount such that far from convergence,
    % the shift is irrelevant and close to convergence, the ratio rho goes
    % to 1, effectively promoting acceptance of the step.
    % The rationale is that close to convergence, both rhonum and rhoden
    % are quadratic in the distance between x and x_prop. Thus, when this
    % distance is on the order of sqrt(eps), the value of rhonum and rhoden
    % is on the order of eps, which is indistinguishable from the numerical
    % error, resulting in badly estimated rho's.
    % For abs(fx) < 1, this heuristic is invariant under offsets of f but
    % not under scaling of f. For abs(fx) > 1, the opposite holds. This
    % should not alarm us, as this heuristic only triggers at the very last
    % iterations if very fine convergence is demanded.
    rho_reg = max(1, abs(fx)) * eps * options.rho_regularization;
    rhonum = rhonum + rho_reg;
    rhoden = rhoden + rho_reg;
   
    if options.debug > 0
        fprintf('DBG:     rhonum : %e\n', rhonum);
        fprintf('DBG:     rhoden : %e\n', rhoden);
    end
    
    % This is always true if a linear, symmetric operator is used for the
    % Hessian (approximation) and if we had infinite numerical precision.
    % In practice, nonlinear approximations of the Hessian such as the
    % built-in finite difference approximation and finite numerical
    % accuracy can cause the model to increase. In such scenarios, we
    % decide to force a rejection of the step and a reduction of the
    % trust-region radius. We test the sign of the regularized rhoden since
    % the regularization is supposed to capture the accuracy to which
    % rhoden is computed: if rhoden were negative before regularization but
    % not after, that should not be (and is not) detected as a failure.
    % 
    % Note (Feb. 17, 2015, NB): the most recent version of tCG already
    % includes a mechanism to ensure model decrease if the Cauchy step
    % attained a decrease (which is theoretically the case under very lax
    % assumptions). This being said, it is always possible that numerical
    % errors will prevent this, so that it is good to keep a safeguard.
    %
    % The current strategy is that, if this should happen, then we reject
    % the step and reduce the trust region radius. This also ensures that
    % the actual cost values are monotonically decreasing.
    %
    % [This bit of code seems not to trigger because tCG already ensures
    %  the model decreases even in the presence of non-linearities; but as
    %  a result the radius is not necessarily decreased. Perhaps we should
    %  change this with the proposed commented line below; needs testing.]
    %
    model_decreased = (rhoden >= 0);
    % model_decreased = (rhoden >= 0) && (stop_inner ~= 6);
    
    if ~model_decreased
        srstr = [srstr ', model did not decrease']; %#ok<AGROW>
    end
    
    rho = rhonum / rhoden;
    
    % Added June 30, 2015 following observation by BM.
    % With this modification, it is guaranteed that a step rejection is
    % always accompanied by a TR reduction. This prevents stagnation in
    % this "corner case" (NaN's really aren't supposed to occur, but it's
    % nice if we can handle them nonetheless).
    if isnan(rho)
        fprintf('rho is NaN! Forcing a radius decrease. This should not happen.\n');
        if isnan(fx_prop)
            fprintf('The cost function returned NaN (perhaps the retraction returned a bad point?)\n');
        else
            fprintf('The cost function did not return a NaN value.\n');
        end
    end
   
    if options.debug > 0
        m = @(x, eta) ...
          getCost(problem, x, storedb, key) + ...
          getDirectionalDerivative(problem, x, eta, storedb, key) + ...
             .5*M.inner(x, getHessian(problem, x, eta, storedb, key), eta);
        zerovec = M.zerovec(x);
        actrho = (fx - fx_prop) / (m(x, zerovec) - m(x, eta));
        fprintf('DBG:   new f(x) : %+e\n', fx_prop);
        fprintf('DBG: actual rho : %e\n', actrho);
        fprintf('DBG:   used rho : %e\n', rho);
    end

    % Choose the new TR radius based on the model performance
    trstr = '   ';
    % If the actual decrease is smaller than 1/4 of the predicted decrease,
    % then reduce the TR radius.
    if rho < 1/4 || ~model_decreased || isnan(rho)
        trstr = 'TR-';
        Delta = Delta/4;
        consecutive_TRplus = 0;
        consecutive_TRminus = consecutive_TRminus + 1;
        if consecutive_TRminus >= 5 && options.verbosity >= 2
            consecutive_TRminus = -inf;
            fprintf(' +++ Detected many consecutive TR- (radius decreases).\n');
            fprintf(' +++ Consider decreasing options.Delta_bar by an order of magnitude.\n');
            fprintf(' +++ Current values: options.Delta_bar = %g and options.Delta0 = %g.\n', options.Delta_bar, options.Delta0);
        end
    % If the actual decrease is at least 3/4 of the predicted decrease and
    % the tCG (inner solve) hit the TR boundary, increase the TR radius.
    % We also keep track of the number of consecutive trust-region radius
    % increases. If there are many, this may indicate the need to adapt the
    % initial and maximum radii.
    elseif rho > 3/4 && (stop_inner == 1 || stop_inner == 2)
        trstr = 'TR+';
        Delta = min(2*Delta, options.Delta_bar);
        consecutive_TRminus = 0;
        consecutive_TRplus = consecutive_TRplus + 1;
        if consecutive_TRplus >= 5 && options.verbosity >= 1
            consecutive_TRplus = -inf;
            fprintf(' +++ Detected many consecutive TR+ (radius increases).\n');
            fprintf(' +++ Consider increasing options.Delta_bar by an order of magnitude.\n');
            fprintf(' +++ Current values: options.Delta_bar = %g and options.Delta0 = %g.\n', options.Delta_bar, options.Delta0);
        end
    else
        % Otherwise, keep the TR radius constant.
        consecutive_TRplus = 0;
        consecutive_TRminus = 0;
    end

    % Choose to accept or reject the proposed step based on the model
    % performance. Note the strict inequality.
    if model_decreased && rho > options.rho_prime
        
        % April 17, 2018: a side effect of rho_regularization > 0 is that
        % it can happen that the cost function appears to go up (although
        % only by a small amount) for some accepted steps. We decide to
        % accept this because, numerically, computing the difference
        % between fx_prop and fx is more difficult than computing the
        % improvement in the model, because fx_prop and fx are on the same
        % order of magnitude yet are separated by a very small gap near
        % convergence, whereas the model improvement is computed as a sum
        % of two small terms. As a result, the step which seems bad may
        % turn out to be good, in that it may help reduce the gradient norm
        % for example. This update merely informs the user of this event.
        % In further updates, we could also introduce this as a stopping
        % criterion. It is then important to choose wisely which of x or
        % x_prop should be returned (perhaps the one with smallest
        % gradient?)
        if fx_prop > fx && options.verbosity >= 2
            fprintf(['Between line above and below, cost function ' ...
                     'increased by %s (step size: %g)\n'], ...
                     fx_prop - fx, norm_eta);
        end
        
        options.accept = true; % passed to next tCG call
        accstr = 'acc';
        % We accept the step: no need to keep the old cache.
        storedb.removefirstifdifferent(key, key_prop);
        x = x_prop;
        key = key_prop;
        fx = fx_prop;
        fgradx = getGradient(problem, x, storedb, key);
        norm_grad = M.norm(x, fgradx);
    else
        % We reject the step: no need to keep cache related to the
        % tentative step.
        storedb.removefirstifdifferent(key_prop, key);
        options.accept = false; % passed to next tCG call
        accstr = 'REJ';
    end
    
    % k is the number of iterations we have accomplished.
    k = k + 1;
    
    % Make sure we don't use too much memory for the store database.
    storedb.purge();

    % Log statistics for freshly executed iteration.
    % Everything after this in the loop is not accounted for in the timing.
    stats = savestats(problem, x, storedb, key, options, k, fx, ...
                      norm_grad, Delta, ticstart, info, rho, rhonum, ...
                      rhoden, options.accept, numit, norm_eta, used_cauchy);
    info(k+1) = stats;

    
    % ** Display:
    if options.verbosity == 2
        fprintf(['%3s %3s   k: %5d     num_inner: %5d     ', ...
        'f: %+e   |grad|: %e   %s\n'], ...
        accstr,trstr,k,numit,fx,norm_grad,srstr);
    elseif options.verbosity > 2
        if options.useRand && used_cauchy
            fprintf('USED CAUCHY POINT\n');
        end
        fprintf('%3s %3s    k: %5d     num_inner: %5d     %s\n', ...
                accstr, trstr, k, numit, srstr);
        fprintf('       f(x) : %+e     |grad| : %e\n',fx,norm_grad);
        if options.debug > 0
            fprintf('      Delta : %f          |eta| : %e\n',Delta,norm_eta);
        end
        fprintf('        rho : %e\n',rho);
    end
    if options.debug > 0
        fprintf('DBG: cos ang(eta,gradf): %d\n',testangle);
        if rho == 0
            fprintf('DBG: rho = 0, this will likely hinder further convergence.\n');
        end
    end

end  % of TR loop (counter: k)

% Restrict info struct-array to useful part
info = info(1:k+1);


if (options.verbosity > 2) || (options.debug > 0)
   fprintf('************************************************************************\n');
end
if (options.verbosity > 0) || (options.debug > 0)
    fprintf('Total time is %f [s] (excludes statsfun)\n', info(end).time);
end

% Return the best cost reached
cost = fx;

end



    

% Routine in charge of collecting the current iteration stats
function stats = savestats(problem, x, storedb, key, options, k, fx, ...
                           norm_grad, Delta, ticstart, info, rho, rhonum, ...
                           rhoden, accept, numit, norm_eta, used_cauchy)
    stats.iter = k;
    stats.cost = fx;
    stats.gradnorm = norm_grad;
    stats.Delta = Delta;
    if k == 0
        stats.time = toc(ticstart);
        stats.rho = inf;
        stats.rhonum = NaN;
        stats.rhoden = NaN;
        stats.accepted = true;
        stats.numinner = 0;
        stats.stepsize = NaN;
        if options.useRand
            stats.cauchy = false;
        end
    else
        stats.time = info(k).time + toc(ticstart);
        stats.rho = rho;
        stats.rhonum = rhonum;
        stats.rhoden = rhoden;
        stats.accepted = accept;
        stats.numinner = numit;
        stats.stepsize = norm_eta;
        if options.useRand
          stats.cauchy = used_cauchy;
        end
    end
    
    % See comment about statsfun above: the x and store passed to statsfun
    % are that of the most recently accepted point after the iteration
    % fully executed.
    stats = applyStatsfun(problem, x, storedb, key, options, stats);
    
end
