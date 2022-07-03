function output = trs_gep(problem, subprobleminput, options, storedb, key)
% Wrapper function for TRSgep which is adapted code based on Satoru Adachi, 
% Satoru Iwata, Yuji Nakatsukasa, and Akiko Takeda's paper which solves the 
% trust region subproblem exactly without iterations.

% minimize <eta,grad> + .5*<eta,Hess(eta)>
% subject to <eta,eta> <= Delta^2
%
% See also: geneigenvalueprob, trs_tCG, trustregions

% This file is part of Manopt: www.manopt.org.
% Original author: Victor Liao, June 13, 2022.
% Contributors: 
% Change log: 
    
    x = subprobleminput.x;
    Delta = subprobleminput.Delta;
    grad = subprobleminput.fgradx;
    
    if options.useRand
        fprintf('options.userand is set to true but trs_gep does not use this option! It will be ignored!\n');
    end

    M = problem.M;
    
    B = tangentorthobasis(M, x);
    H = hessianmatrix(problem, x, B);
    
    n = size(H,1);

    grad_vec = tangent2vec(M, x, B, grad);

    [eta_vec, limitedbyTR] = geneigenvalueprob(H, grad_vec, Delta);

    eta = lincomb(M, x, B, eta_vec);
    
    Heta_vec = H*eta_vec;
    Heta = lincomb(M, x, B, Heta_vec);

    output.eta = eta;
    output.Heta = Heta;
    output.numit = problem.M.dim();
    output.limitedbyTR = limitedbyTR;
    if limitedbyTR
        output.stopreason_str = 'Exact trs_gep boundary sol';
    else
        output.stopreason_str = 'Exact trs_gep interior sol';
    end
