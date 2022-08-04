function [eta, Heta, print_str, stats] = trs_gep(problem, subprobleminput, options, storedb, key)
% Solves trust-region subproblem with TRSgep in a subspace of tangent space.
%
% minimize <eta,grad> + .5*<eta,Hess(eta)>
% subject to <eta,eta> <= Delta^2
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
% Example to solve trust-region subproblem restricted to two dimensional
% subspace (assuming M.dim() >= 2):
% options.subproblemsolver = @trs_gep;
% options.gepsubspacedim = 2;
%
% See also: TRSgep trs_tCG trustregions

% This file is part of Manopt: www.manopt.org.
% Original author: Victor Liao, June 13, 2022.
% Contributors: 
% Change log: 

    if nargin == 1
        % Only problem passed in signals that trustregions.m wants default
        % values for stats.
        eta = problem.M.zerovec();
        Heta = problem.M.zerovec();
        print_str = '';
        stats = struct('hessvecevals', 0, 'limitedbyTR', false, ...
                    'memorytCG_MB', 0);
        return;
    end

    x = subprobleminput.x;
    Delta = subprobleminput.Delta;
    grad = subprobleminput.fgradx;
    
    print_str = '';

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
        print_str = sprintf('hessvecevals: %5d     %s', n, stopreason_str);
    elseif options.verbosity > 2
        print_str = sprintf('\nhessvecevals: %5d     %s', n, stopreason_str);
    end
    stats = struct('hessvecevals', n, 'limitedbyTR', limitedbyTR, ...
                    'memorytCG_MB', memorytCG_MB);


