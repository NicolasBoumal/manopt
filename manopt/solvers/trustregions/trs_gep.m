function [eta, Heta, print_str, stats] = trs_gep(problem, subprobleminput, options, storedb, key)
% Solves trust-region subproblem with TRSgep in a subspace of tangent space.
%
% minimize <eta,grad> + .5*<eta,Hess(eta)>
% subject to <eta,eta> <= Delta^2
%
% function [~, ~, ~, stats] = trs_gep()
% function [eta, Heta, print_str, stats] = trs_gep(problem, subprobleminput, options, storedb, key)
%
% If options.gepsubspacedim = M.dim() then trs_gep solves the trust-region 
% subproblem exactly in the entire tangent space by creating an orthonormal 
% basis for the entire subspace using tangentorthobasis, then passing the 
% Hessian and gradient in that basis to TRSgep.
% 
% If options.gepsubspacedim < M.dim() then we solve the trust-region
% subproblem restricted to a subspace of dimension options.gepsubspacedim.
% To do this, we obtain an orthonormal basis of a options.gepsubspacedim 
% dimensional subspace and pass the Hessian and gradient in that basis to
% TRSgep.
% 
% When grad is nonzero, to guarantee that we do at least as well as the 
% Cauchy point, we ensure grad is in the span of the basis.
%
% When grad is zero, then our basis is constructed using a randomly sampled
% set of linearly independent vectors and orthonormalized with
% Gram-Schmidt. This can, in principle, help in escaping saddle points.
%
% Inputs:
%   problem: Manopt optimization problem structure
%   subprobleminput: struct storing information for this subproblemsolver
%       x: point on the manifold problem.M
%       grad: gradient of the cost function of the problem at x
%       eta: starting point problem.M.zerovec(x) or small random tangent
%       vector if options.useRand == true.
%       Delta = trust-region radius
%   options: structure containing options for the subproblem solver
%   storedb, key: caching data for problem at x
%
% Options specific to this subproblem solver:
%   gepsubspacedim (problem.M.dim())
%       A value between 1 and problem.M.dim() inclusive that specifies the
%       subspace dimension we wish to solve the trust-region subproblem
%       over.
%       
% Outputs:
%   eta: approximate solution to the trust-region subproblem at x
%   Heta: Hess f(x)[eta] -- this is necessary in the outer loop, and it
%       is often naturally available to the subproblem solver at the
%       end of execution, so that it may be cheaper to return it here.
%   print_str: subproblem specific string to be printed by trustregions.m
%   stats: structure with values to be stored in trustregions.m
%       hessvecevals: number of Hessian calls during execution (here we set
%       it to gepsubspacedim since we construct the Hessian
%       explicitly in gepsubspacedim dimensions.
%       limitedbyTR: true if a boundary solution is returned
%
% Note: If nargin == 0, then the returned stats struct will contain the 
% relevant fields along with their corresponding initial values. print_str 
% will also contain the header to be printed before the first pass of 
% trustregions.m (if options.verbosity == 2). The other outputs will be 
% empty. This stats struct is used in the first call to savestats in 
% trustregions.m to initialize the info struct properly.
%
% Note: trs_gep does not use preconditioning.
%
% Example to solve trust-region subproblem restricted to two dimensional
% subspace (assuming M.dim() >= 2) where if grad != 0, the subspace spans 
% grad and one random linearly independent tangent vector, otherwise the
% subspace spans a completely randomized two dimensional subspace:
%
% options.subproblemsolver = @trs_gep;
% options.gepsubspacedim = 2;
%
% See also: TRSgep trs_tCG trustregions

% This file is part of Manopt: www.manopt.org.
% Original author: Victor Liao, June 13, 2022.
% Contributors: 
% Change log: 

    if nargin == 0
        % trustregions.m only wants default values for stats.
        eta = [];
        Heta = [];
        print_str = sprintf('%-13s%s\n','hessvecevals', 'stopreason');
        stats = struct('hessvecevals', 0, 'limitedbyTR', false);
        return;
    end

    x = subprobleminput.x;
    Delta = subprobleminput.Delta;
    grad = subprobleminput.fgradx;
    
    if options.useRand	
        warning('manopt:trs_gep:useRand', ...
        ['(options.useRand == true) but trs_gep does not use this option. It will be ignored.' ...
        'To disable this warning: warning(''off'', ''manopt:trs_gep:useRand'')']);
    end
    
    M = problem.M;

    % Set local defaults here    
    localdefaults.gepsubspacedim = M.dim();

    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    % dimension of subspace we want to solve TRS over.
    n = options.gepsubspacedim;
    
    assert(n >= 1 && n <= M.dim() && n == round(n), ...
           'options.gepsubspacedim must be an integer between 1 and M.dim().');

    % If gradient is nonzero, then even if n < M.dim() we
    % guarantee to do at least as well as the cauchy point.
    grad_norm = M.norm(x, grad);
    if ~(grad_norm == 0)
        basis_vecs = cell(1, 1);
        basis_vecs{1} = grad;

        B = tangentorthobasis(M, x, n, basis_vecs);

        grad_vec = zeros(n, 1);
        grad_vec(1) = grad_norm;
    else
        B = tangentorthobasis(M, x, n);

        grad_vec = tangent2vec(M, x, B, grad);
    end
    
    H = hessianmatrix(problem, x, B);

    [eta_vec, limitedbyTR] = TRSgep(H, grad_vec, Delta);

    eta = lincomb(M, x, B, eta_vec);
    
    % Instead of the below two lines one can simply use: Heta =
    % getHessian(problem, x, eta, storedb, key). This will be problem
    % dependent and can be reconsidered depending on application.
    Heta_vec = H*eta_vec;
    Heta = lincomb(M, x, B, Heta_vec);

    if limitedbyTR
        stopreason_str = 'Exact trs_gep boundary sol';
    else
        stopreason_str = 'Exact trs_gep interior sol';
    end

    if options.verbosity == 2
        print_str = sprintf('    %-5d        %s', n, stopreason_str);
    elseif options.verbosity == 3
        print_str = sprintf('hessvecevals: %5d  %s', n, stopreason_str);
    elseif options.verbosity > 3
        print_str = sprintf('\nhessvecevals: %5d   %s', n, stopreason_str);
    end
    stats = struct('hessvecevals', n, 'limitedbyTR', limitedbyTR);


