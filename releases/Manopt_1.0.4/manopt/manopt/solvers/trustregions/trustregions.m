function [x cost info] = trustregions(problem, x, options)
% Riemannian trust-regions solver for optimization on manifolds.
%
% function [x cost info] = trustregions(problem)
% function [x cost info] = trustregions(problem, x0)
% function [x cost info] = trustregions(problem, x0, options)
% function [x cost info] = trustregions(problem, [], options)
%
% RTR Riemannian Trust-Region (with tCG inner solve). This solver will
% attempt to minimize the cost function described in the problem structure.
% It requires the availability of the cost function and of its gradient. It
% will issue calls for the Hessian (if no Hessian nor approximate Hessian
% is provided, a standard approximation of the Hessian based on the
% gradient will be computed). If a preconditioner for the Hessian is
% provided, it will be used.
%
% The initial iterate is x0 if it is provided (otherwise, a random point on
% the manifold is picked). To specify options whilst not specifying an
% initial iterate, give x0 as [] (the empty matrix).
%
% None of the options are mandatory. See the documentation for details.
%
% For a description of the algorithm and theorems offering convergence
% guarantees, see the references below.
%
% For stopping criteria (maxtime, maxiter, tolgradnorm... in options
% struct), see http://www.manopt.org/tutorial.html#solvers as well as this
% solver's specific documentation:
% http://www.manopt.org/solver_documentation_trustregions.html
%
% Specific options:
%   options.Delta0    - Initial trust-region radius.
%   options.Delta_bar - Maximum trust-region radius.
%   options.miniter   - Minimum number of outer iterations
%                       (default: 0); used only with randomization
%   options.maxiter   - Maximum number of outer iterations
%                       (default: 1000)
%   options.mininner  - Minimum number of inner iterations
%                       (default: 0).
%   options.maxinner  - Maximum number of inner iterations
%                       (default: dimension of manifold)
%   options.useRand   - Set to true if the trust-region solve is to be
%                       initiated with a random tangent vector. If set to
%                       true, no preconditioner will be used.
%                       (default: false)
%   options.kappa     - Inner kappa convergence tolerance
%   options.theta     - Inner theta convergence tolerance
%   options.rho_prime - Accept/reject ratio
%
% The info output is a struct array, with fields:
%  iter - the outer iteration number for the stats
%  gradnorm - the norm of the gradient, sqrt(g(x,gradfx,gradfx))
%  cost - the current value under the objective function
%  rho - the performance ratio for the iterate
%  time - the total time in seconds to reach the corresponding cost
%  accepted - whether the proposed iterate was accepted or not
%  numinner - the # of inner iterations used to compute the next iterate
%  Delta - the trust-region radius at the outer iteration
%  cauchy - whether the Cauchy point was used or not (if useRand is true)
% 

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
%  NB April 3, 2013: tCG now return the Hessian along the returned
%                    direction eta, so that we do not compute that Hessian
%                    redundantly: some savings at each iteration.
%                    Similarly, if the useRand flag is on, we spare an
%                    extra Hessian computation at each out iteration too
%                    thanks to some modifications in the Cauchy point
%                    section of the useRand-selective code.
%
%  NB Aug. 22, 2013: Made some changes so that this is now Octave
%                    compatible: tic/toc pair used as is, without the
%                    (safer) Matlab way of doing it:
%                     t = tic(); elapsed = toc(t);
%                     savestats was moves to not be a nested function
%                     anymore (and this is ugly, but necessary for Octave
%                     compatibility).


% Verify that the problem description is sufficient for the solver.
if ~canGetCost(problem)
    warning('manopt:getCost', ...
            'No cost provided. The algorithm will likely abort.');  
end
if ~canGetGradient(problem)
    warning('manopt:getGradient', ...
            'No gradient provided. The algorithm will likely abort.');    
end
if ~canGetHessian(problem)
    warning('manopt:getHessian:approx', ...
            'No Hessian provided. Using an approximation instead.');
end

% Define some strings for display
tcg_stop_reason = {'negative curvature',...
                   'exceeded trust region',...
                   'reached target residual-kappa',...
                   'reached target residual-theta',...
                   'dimension exceeded'};

% Set local defaults here
localdefaults.verbosity = 2;
localdefaults.maxtime = inf;
localdefaults.miniter = 3;
localdefaults.maxiter = 1000;
localdefaults.mininner = 1;
localdefaults.maxinner = problem.M.dim();
localdefaults.tolgradnorm = 1e-6;
localdefaults.kappa = 0.1;
localdefaults.theta = 1.0;
localdefaults.rho_prime = 0.1;
localdefaults.kappa_easy = 0.001;
localdefaults.useRand = false;
if isfield('typicaldist', problem.M)
    localdefaults.Delta_bar = problem.M.typicaldist();
else
    localdefaults.Delta_bar = sqrt(problem.M.dim());
end 
localdefaults.Delta0 = localdefaults.Delta_bar / 8;

% Merge global and local defaults, then merge w/ user options, if any.
localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
if ~exist('options', 'var') || isempty(options)
    options = struct();
end
options = mergeOptions(localdefaults, options);

if options.debug
    disp(options);
end

% Create a store database
storedb = struct();

tic();

% If no initial point x is given by the user, generate one at random.
if ~exist('x', 'var') || isempty(x)
    x = problem.M.rand();
end

%% Initializations

% k counts the outer (TR) iterations. The semantic is that k counts the
% number of iterations fully executed so far.
k = 0;

% initialize solution and companion measures: f(x), fgrad(x)
[fx fgradx storedb] = getCostGrad(problem, x, storedb);
norm_grad = problem.M.norm(x, fgradx);

% initialize trust-region radius
Delta = options.Delta0;

% Save stats in a struct array info, and preallocate
% (see http://people.csail.mit.edu/jskelly/blog/?x=entry:entry091030-033941)
if ~exist('used_cauchy', 'var')
    used_cauchy = [];
end
stats = savestats(problem, x, storedb, options, k, fx, norm_grad, Delta);
info(1) = stats;
info(min(10000, options.maxiter+1)).iter = [];

% ** Display:
if options.verbosity == 2
   fprintf(['%3s %3s      %5s                %5s     ',...
            'f: %e   |grad|: %e\n'],...
           '   ','   ','     ','     ', fx, norm_grad);
elseif options.verbosity > 2
   fprintf('************************************************************************\n');
   fprintf('%3s %3s    k: %5s     num_inner: %5s     %s\n',...
           '','','______','______','');
   fprintf('       f(x) : %e       |grad| : %e\n', fx, norm_grad);
   fprintf('      Delta : %f\n', Delta);
end


% **********************
% ** Start of TR loop **
% **********************
while true
    
	% Start clock for this outer iteration
    tic();

    % Run standard stopping criterion checks
    [stop reason] = stoppingcriterion(problem, x, options, info, k+1);
    
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
        eta = problem.M.zerovec(x);
    else
        % Random vector in T_x M (this has to be very small)
        eta = problem.M.lincomb(x, 1e-6, problem.M.randvec(x));
        % Must be inside trust-region
        while problem.M.norm(x, eta) > Delta
            eta = problem.M.lincomb(x, sqrt(sqrt(eps)), eta);
        end
    end

    % solve TR subproblem
    [eta Heta numit stop_inner storedb] = ...
                     tCG(problem, x, fgradx, eta, Delta, options, storedb);
    srstr = tcg_stop_reason{stop_inner};
    norm_eta = problem.M.norm(x, eta);
    if options.debug > 0
        testangle = problem.M.inner(x, eta, fgradx) / (norm_eta*norm_grad);
    end

    % If using randomized approach, compare result with the Cauchy point.
    % Convergence proofs assume that we achieve at least the reduction of
    % the Cauchy point. After this if-block, either all eta-related
    % quantities have been changed consistently, or none of them have
    % changed.
    if options.useRand
        used_cauchy = false;
        % Check the curvature,
        [Hg storedb] = getHessian(problem, x, fgradx, storedb);
        g_Hg = problem.M.inner(x, fgradx, Hg);
        if g_Hg <= 0
            tau_c = 1;
        else
            tau_c = min( norm_grad^3/(Delta*g_Hg) , 1);
        end
        % and generate the Cauchy point.
        eta_c  = problem.M.lincomb(x, -tau_c * Delta / norm_grad, fgradx);
        Heta_c = problem.M.lincomb(x, -tau_c * Delta / norm_grad, Hg);
        % [Heta_c storedb] = getHessian(problem, x, eta_c, storedb);

        % Now that we have computed the Cauchy point in addition to the
        % returned eta, we might as well keep the best of them.
        mdle  = fx + problem.M.inner(x, fgradx, eta) ...
                   + .5*problem.M.inner(x, Heta,   eta);
        mdlec = fx + problem.M.inner(x, fgradx, eta_c) ...
                   + .5*problem.M.inner(x, Heta_c, eta_c);
        if mdle > mdlec
            eta = eta_c;
            Heta = Heta_c; % added April 11, 2012
            norm_eta = problem.M.norm(x, eta);
            used_cauchy = true;
        end
    end 

	% Compute the retraction of the proposal
	x_prop  = problem.M.retr(x, eta);

	% Compute function value of the proposal
	[fx_prop storedb] = getCost(problem, x_prop, storedb);

	% Will we accept the proposed solution or not?
    % Check the performance of the quadratic model against the actual cost.
    rhonum = fx - fx_prop;
    rhoden = -problem.M.inner(x, fgradx, eta) ...
             -.5*problem.M.inner(x, eta, Heta);
   
    if options.debug > 0
        fprintf('DBG:     rhonum : %e\n', rhonum);
        fprintf('DBG:     rhoden : %e\n', rhoden);
    end
    
    model_did_not_decrease = (rhoden < 0);
    
    % This if-block should never trigger. It can though, when something is
    % wrong with the quadratic model (that is, with the gradient and the
    % Hessian or its approximation). It could simply be numerical accuracy
    % problems, but it could also be that the Hessian (or its
    % approximation) is not linear and / or not symmetric. If we don't do
    % anything about it, the algorithm as presently implemented could make
    % an upwards step, letting the cost increase. That is usually not
    % desirable, but I'm not sure how to deal with this in general.
    if model_did_not_decrease
        
% % %         if options.verbosity >= 2
% % %             fprintf('No model decrease: something is wrong.\n');
% % %             fprintf('This typically happens if the Hessian or its approximation\n');
% % %             fprintf('is not symmetric or, worse, not linear. In particular,\n');
% % %             fprintf('the automatic finite differences approximation of the\n');
% % %             fprintf('Hessian is not linear, and may explain this failure.\n');
% % %         end
        
        % Get ready for damage control. The remainder of the code assumes 
        % that rhoden >= 0, and hence that rho >= 0 means the actual cost
        % is decreasing (or, possibly, remaining constant). Of course,
        % if rhoden < 0, then that isn't true anymore, and the algorithm
        % could accept a step that makes the cost increase. In order to
        % make sure that the algorithm is a descent method, we simply give
        % up ... 
        
        % TODO: if the actual cost decreased anyway (rhonum > 0), we really
        % should accept that step ...
        
        % TODO: I tried going for the Cauchy point (looked like something
        % could come out of it) as well as detecting in tCG directly if
        % something wrong is happening (I tried to look for increasing cost
        % of the subproblem, but you could also monitor the norm of the
        % residue, which is supposed to decrease monotonically). Perhaps
        % something a bit more clever could be done at that level.
        
% % %         if options.verbosity >= 1
% % %             fprintf(['The quadratic model could not be decreased ' ...
% % %                      '(is the approximate Hessian linear and symmetric?).\n']);
% % %             fprintf('Try using a larger options.tolgradnorm value.\n');
% % %         end
        
        % TODO : Should break to stop here, but we need to log all that
        % happened until now honestly; can't hide the many Hessian
        % evaluations that we're about to just discard!
        % break;
        
    end
   
    rho = rhonum / rhoden;
   
    if options.debug > 1,
        m = @(x, eta) ...
          getCost(problem, x, storedb) + ...
          getDirectionalDerivative(problem, x, eta, storedb) + ...
          .5*problem.M.inner(x, getHessian(problem, x, eta, storedb), eta);
        zerovec = problem.M.zerovec();
        actrho = (fx - fx_prop) / (m(x, zerovec) - m(x, eta));
        fprintf('DBG:   new f(x) : %e\n', fx_prop);
        fprintf('DBG: actual rho : %e\n', actrho);
        fprintf('DBG:   used rho : %e\n', rho);
    end
   
    % HEURISTIC WARNING:
    % If abs(model change) is relatively zero, we are probably near a
    % critical point. Set rho to 1.
    % NB July 12, 2013: this seems fairly important in practice, near
    % convergence anyway. Might be worthwhile to investigate this, perhaps
    % improve it (the division by fx bugs me a lot), and certainly document
    % it...
    if abs(rhonum/fx) < sqrt(eps),
        if options.debug
            fprintf(' *** HEURISTIC TRIGGERED\n');
        end
        small_rhonum = rhonum;
        rho = 1;
    else 
        small_rhonum = 0;
    end

    % choose new TR radius based on performance
    trstr = '   ';
    if rho < 1/4
        trstr = 'TR-';
        %Delta = 1/4*norm_eta;
        Delta = Delta/4;
    elseif rho > 3/4 && (stop_inner == 2 || stop_inner == 1),
        trstr = 'TR+';
        %Delta = min(2*norm_eta,Delta_bar);
        Delta = min(2*Delta, options.Delta_bar);
    end

    % Choose new iterate based on performance
    if rho > options.rho_prime
        accept = true;
        accstr = 'acc';
        x = x_prop;
        fx = fx_prop;
        [fgradx storedb] = getGradient(problem, x, storedb);
        norm_grad = problem.M.norm(x, fgradx);
    else
        accept = false;
        accstr = 'REJ';
    end
    
    
    % Make sure we don't use too much memory for the store database
    storedb = purgeStoredb(storedb, options.storedepth);
    
    % k is the number of iterations we have accomplished.
    k = k + 1;

    % Log statistics for freshly executed iteration.
    % Everything after this in the loop is not accounted for in the timing.
    stats = savestats(problem, x, storedb, options, k, fx, norm_grad, ...
                      Delta, info, rho, rhonum, rhoden, accept, numit, ...
                      used_cauchy);
    info(k+1) = stats; %#ok<AGROW>

    
    % ** Display:
    if options.verbosity == 2,
        fprintf(['%3s %3s   k: %5d     num_inner: %5d     ', ...
        'f: %e   |grad|: %e   %s\n'], ...
        accstr,trstr,k,numit,fx,norm_grad,srstr);
    elseif options.verbosity > 2,
        if options.useRand && used_cauchy,
            fprintf('USED CAUCHY POINT\n');
        end
            fprintf('%3s %3s    k: %5d     num_inner: %5d     %s\n', ...
                    accstr, trstr, k, numit, srstr);
            fprintf('       f(x) : %e     |grad| : %e\n',fx,norm_grad);
            fprintf('      Delta : %f          |eta| : %e\n',Delta,norm_eta);
        if small_rhonum ~= 0,
            fprintf('VERY SMALL rho_num: %e\n',small_rhonum);
        else
            fprintf('        rho : %e\n',rho);
        end
    end
    if options.debug > 0,
        fprintf('DBG: cos ang(eta,gradf): %d\n',testangle);
        if rho == 0
            keyboard;
        end
    end

end  % of TR loop (counter: k)

% Restrict info struct-array to useful part
info = info(1:k+1);


if (options.verbosity > 2) || (options.debug > 0),
   fprintf('************************************************************************\n');
end
if (options.verbosity > 0) || (options.debug > 0)
    fprintf('Total time is %f [s] (excludes statsfun)\n', info(end).time);
end

% Return the best cost reached
cost = fx;

end



    

% Routine in charge of collecting the current iteration stats
function stats = savestats(problem, x, storedb, options, k, fx, ...
                           norm_grad, Delta, info, rho, rhonum, ...
                           rhoden, accept, numit, used_cauchy)
    stats.iter = k;
    stats.cost = fx;
    stats.gradnorm = norm_grad;
    stats.Delta = Delta;
    if k == 0
        stats.time = toc();
        stats.rho = inf;
        stats.rhonum = 0;
        stats.rhoden = 0;
        stats.accepted = true;
        stats.numinner = 0;
        if options.useRand
            stats.cauchy = false;
        end
    else
        stats.time = info(k).time + toc();
        stats.rho = rho;
        stats.rhonum = rhonum;
        stats.rhoden = rhoden;
        stats.accepted = accept;
        stats.numinner = numit;
        if options.useRand,
          stats.cauchy = used_cauchy;
        end
    end
    
    % TODO : resolve which x (and thus which store) should be passed to the
    % applyStatsfun function. In the present situation, if the step is
    % accepted, the nex x is passed and hence the new store is passed. The
    % former store, which contains all the info the user might have logged
    % about the work needed to get to the new x, is invisible :/.
    % Conversely, if the step is rejected more than once, the work-counters
    % that the user maintains in the store structure are incremented,
    % logged (step rejected) further incremented and re-logged, thus making
    % it look like the work before the first reject was done twice. This
    % isn't right either.
    % It might be ok somehow (after all, you get the latest point reached,
    % and the store structure that goes with it), but then the
    % documentation should be clear about the risks of trying to count
    % things with it, and give alternatives. For example, you can have a
    % counter variable that is accessible from cost, grad, statsfun etc and
    % then rely on the fact that statsfun is called exactly once per
    % iteration, no matter the details.
    stats = applyStatsfun(problem, x, storedb, options, stats);
end
