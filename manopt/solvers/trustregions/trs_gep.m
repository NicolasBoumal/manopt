function output = trs_gep(problem, subprobleminput, options, storedb, key)
% Wrapper function for TRSgep which is adapted code based on Satoru Adachi, 
% Satoru Iwata, Yuji Nakatsukasa, and Akiko Takeda's paper which solves the 
% trust region subproblem exactly without iterations.

% If gepdim (see below) is specified to be smaller than problem.M.dim(),
% expect a large speedup in computation and improvement is still guaranteed
% to be at least as good as the cauchy point.

% minimize <eta,grad> + .5*<eta,Hess(eta)>
% subject to <eta,eta> <= Delta^2
%
% The options structure is used to overwrite the default values. All
% options have a default value and are hence optional. To force an option
% value, pass an options structure with a field options.optionname, where
% optionname is one of the following and the default value is indicated
% between parentheses:
%
%   gepdim (problem.M.dim())
%       The number of orthonormal vectors in the basis returned by
%       tangentorthobasis. If we take gepdim < problem.M.dim() then we add 
%       grad to ensure grad is in the span of the basis. Thus, TRSgep is
%       guaranteed to do at least as well as the cauchy point. If grad is
%       zero then we simply orthonormalize a random set of gepdim number of
%       tangent vectors.

% See also: TRSgep, trs_tCG, trustregions

% This file is part of Manopt: www.manopt.org.
% Original author: Victor Liao, June 13, 2022.
% Contributors: 
% Change log: 
    
    x = subprobleminput.x;
    Delta = subprobleminput.Delta;
    grad = subprobleminput.fgradx;
    
    if options.useRand	
        fprintf('options.useRand is set to true but trs_gep does not use this option! It will be ignored!\n');	
    end
    
    M = problem.M;

    % Set local defaults here    
    localdefaults.gepdim = M.dim();

    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    % If we use lower dimensional subspace ensure it contains the gradient
    % vector to ensure we do at least as well as the cauchy point.
    if options.gepdim < M.dim() && ~isequal(grad, M.zerovec())
        basis_vecs = cell(1, 1);
        basis_vecs{1} = grad;
        B = tangentorthobasis(M, x, options.gepdim, basis_vecs);
    else
        B = tangentorthobasis(M, x, options.gepdim);
    end
    
    H = hessianmatrix(problem, x, B);

    grad_vec = tangent2vec(M, x, B, grad);

    [eta_vec, limitedbyTR] = TRSgep(H, grad_vec, Delta);

    eta = lincomb(M, x, B, eta_vec);
    
    Heta_vec = H*eta_vec;
    Heta = lincomb(M, x, B, Heta_vec);

    output.eta = eta;
    output.Heta = Heta;
    output.numit = options.gepdim;
    output.limitedbyTR = limitedbyTR;
    if limitedbyTR
        output.stopreason_str = 'Exact trs_gep boundary sol';
    else
        output.stopreason_str = 'Exact trs_gep interior sol';
    end
