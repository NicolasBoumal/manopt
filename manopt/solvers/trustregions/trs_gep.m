function [eta, Heta, print_str, stats] = trs_gep(problem, subprobleminput, options, ~, ~)
% Solves trust-region subproblem with TRSgep in a subspace of tangent space.
%
% minimize <eta,grad> + .5*<eta,Hess(eta)>
% subject to <eta,eta> <= Delta^2
%
% function [eta, Heta, print_str, stats] = trs_gep(problem, subprobleminput, options, ~, ~)
%
% This is meant to be used by the trustregion solver.
% To use this method, specify trs_gep as an option, and your chosen
% subspace dimension in the problem structure, as follows:
%
% n = problem.M.dim()
% options.subproblemsolver = @trs_gep;
% options.gepsubspacedim = n; % any integer between 1 and n inclusive.
% x = trustregions(problem, [], options)
%
% Note: trs_gep does not use preconditioning.
%
% trs_gep solves the trust-region subproblem exactly in a 
% options.gepsubspacedim dimensional subspace of the tangent space by 
% creating an orthonormal basis for the subspace using tangentorthobasis, 
% and passing the hessian and gradient in that basis to TRSgep.
% 
% The basis is constructed by tangentorthobasis with randomly sampled 
% linearly independent tangent vectors then orthonormalized with 
% Gram-Schmidt.
%
% If options.gepsubspacedim < problem.M.dim() then when grad is nonzero, 
% to guarantee that we do at least as well as the 
% Cauchy point, we ensure grad is in the span of the basis.
%
% Inputs:
%   problem: Manopt optimization problem structure
%   subprobleminput: struct storing information for this subproblemsolver
%       x: point on the manifold problem.M
%       grad: gradient of the cost function of the problem at x
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
% trs_gep can also be called in the following way (for printing
% purposes):
%
% function [~, ~, print_str, stats] = trs_gep([], [], options)
%
% In this case when nargin == 3, the returned stats struct contains the 
% relevant fields along with their corresponding initial values. In this
% case print_str is the header to be printed before the first pass of 
% trustregions.m. The other outputs will be 
% empty. This stats struct is used in the first call to savestats in 
% trustregions.m to initialize the info struct properly.
%
% See also: TRSgep trs_tCG trustregions

% This file is part of Manopt: www.manopt.org.
% Original author: Victor Liao, June 13, 2022.
% Contributors: 
% Change log: 

    if nargin == 3
        % trustregions.m only wants default values for stats.
        eta = [];
        Heta = [];
        print_str = sprintf('%9s   %s', 'hessvec','stopreason');
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
    print_str = sprintf('%9d   %s', n, stopreason_str);
    stats = struct('hessvecevals', n, 'limitedbyTR', limitedbyTR);


