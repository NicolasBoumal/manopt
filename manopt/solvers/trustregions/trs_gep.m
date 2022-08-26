function trsoutput = trs_gep(problem, trsinput, options, ~, ~)
% Solves trust-region subproblem with TRSgep in a subspace of tangent space.
% 
% function trsoutput = trs_gep(problem, trsinput, options, storedb, key)
% 
% minimize <eta, grad> + .5*<eta, Hess(eta)>
% subject to <eta, eta> <= Delta^2
%
% This is meant to be used by the trustregion solver.
% To use this method, specify trs_gep as an option, and your chosen
% subspace dimension in the problem structure, as follows:
%
%   options.subproblemsolver = @trs_gep;
%   options.gepsubspacedim = n; % Integer in the range 1:problem.M.dim().
%                               % If omitted, default is problem.M.dim().
%   x = trustregions(problem, [], options);
%
% Note: trs_gep does not use preconditioning.
%
% In principle, trs_gep solves the trust-region subproblem exactly in a 
% subspace of the tangent space with dimension options.gepsubspacedim.
% If that dimension is equal to the manifold dimension, then the solver
% is meant to find a global optimum of the TRS, up to numerical issues.
%
% This function achieves that goal as follows: it creates an orthonormal
% basis for (a subspace of) the tangent space using tangentorthobasis, 
% it expresses the Hessian and gradient of the cost function at the
% current point x (restricted to the subspace) in the chosen basis, and it
% passes those objects to TRSgep.
% 
% The basis is constructed by tangentorthobasis with randomly tangent
% vectors (linearly independent with probability 1) then orthonormalized.
% Therefore, if gepsubspacedim is less than the manifold dimension, the
% minimization is executed over a random subspace. In that scenario, if the
% gradient is nonzero, the gradient is included in the basis to be
% orthonormalized. This ensures that the point returned by this solver is
% always as good as the Cauchy point.
%
% Constructing the basis itself can be time consuming in high dimensions,
% and aiming for an exact solve of the TRS as well. This subproblem solver
% is meant mostly for research, not for efficiency.
%
%
% Inputs:
%   problem: Manopt optimization problem structure
%   trsinput: structure with the following fields:
%       x: point on the manifold problem.M
%       fgradx: gradient of the cost function of the problem at x
%       Delta: trust-region radius
%   options: structure containing options for the subproblem solver
%   storedb, key: manopt's caching system for the point x
%
% Options specific to this subproblem solver (default value):
%   gepsubspacedim (problem.M.dim())
%       A value between 1 and problem.M.dim() inclusive that specifies the
%       dimension of the subpsace over which we wish to solve the
%       trust-region subproblem.
%       
% Output: the structure trsoutput contains the following fields:
%       eta: approximate solution to the trust-region subproblem at x
%       Heta: Hess f(x)[eta] -- this is necessary in the outer loop, and it
%           is often naturally available to the subproblem solver at the
%           end of execution, so that it may be cheaper to return it here.
%       limitedbyTR: true if a boundary solution is returned
%       printstr: logged information to be printed by trustregions.
%       stats: structure with the following statistics:
%           hessvecevals: number of Hessian-vector calls issued
%
%
% trs_gep can also be called in the following way (by trustregions) to
% obtain part of the header to print and an initial stats structure:
%
% function trsoutput = trs_gep([], [], options)
%
% In this case, trsoutput contains the following fields:
%   printheader: subproblem header to be printed before the first loop of 
%       trustregions
%   initstats: struct with initial values for stored stats in subsequent
%       calls to trs_gep. Used in the first call to savestats 
%       in trustregions to initialize the info struct properly.
%
% See also: TRSgep trs_tCG trustregions

% This file is part of Manopt: www.manopt.org.
% Original author: Victor Liao, June 13, 2022.
% Contributors: Nicolas Boumal
% Change log: 

    % trustregions only wants header and default values for stats.
    if nargin == 3 && isempty(problem) && isempty(trsinput)
        trsoutput.printheader = sprintf('%9s   %s', 'hessvec', 'stopreason');
        trsoutput.initstats = struct('hessvecevals', 0);
        return;
    end

    x = trsinput.x;
    grad = trsinput.fgradx;
    Delta = trsinput.Delta;
    
    M = problem.M;

    % Set local defaults here
    localdefaults.gepsubspacedim = M.dim();

    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    % dimension of subspace over which to solve the TRS.
    n = options.gepsubspacedim;
    
    assert(n >= 1 && n <= M.dim() && n == round(n), ...
       'options.gepsubspacedim must be an integer between 1 and M.dim().');

    % If gradient is nonzero, then even if n < M.dim() we
    % guarantee to do at least as well as the cauchy point
    % by including the gradient in the basis of the subspace.
    % The column vector grad_vec contains the coordinates of grad in the
    % basis B.
    grad_norm = M.norm(x, grad);
    if grad_norm ~= 0
        % Append n-1 random tangent vectors to the gradient,
        % then orthonormalize with Gram-Schmidt.
        B = tangentorthobasis(M, x, n, {grad});
        grad_vec = zeros(n, 1);
        grad_vec(1) = grad_norm;
    else
        B = tangentorthobasis(M, x, n);
        grad_vec = zeros(n, 1);
    end
    
    % Express the Hessian of the cost function f at x in the basis B.
    % If B is a basis for a subspace of T_x M rather than for the whole
    % tangent space, then H represents the Hessian restriced to that
    % subspace.
    H = hessianmatrix(problem, x, B);

    % This is where the actual work happens.
    [eta_vec, limitedbyTR] = TRSgep(H, grad_vec, Delta);

    % Construct the tangent vector eta using its coordinates eta_vec in the
    % basis B.
    eta = lincomb(M, x, B, eta_vec);
    
    % We want to return Heta, defined by:
    %   Heta = getHessian(problem, x, eta, storedb, key).
    % This however requires one Hessien-vector call.
    % We can avoid issuing that cal with the two lines below.
    % This is likely to be faster, but may be less accurate numerically.
    % Which is better may depend on the application.
    Heta_vec = H*eta_vec;
    Heta = lincomb(M, x, B, Heta_vec);

    if limitedbyTR
        stopreason_str = 'Exact trs_gep boundary sol';
    else
        stopreason_str = 'Exact trs_gep interior sol';
    end
    
    printstr = sprintf('%9d   %s', n, stopreason_str);
    stats = struct('hessvecevals', n);
    
    trsoutput.eta = eta;
    trsoutput.Heta = Heta;
    trsoutput.limitedbyTR = limitedbyTR;
    trsoutput.printstr = printstr;
    trsoutput.stats = stats;
end
