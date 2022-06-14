function [eta, Heta, numit, stop_inner]= TRSgep(problem, x, grad, ~, Delta, ~, ~, ~)
% Wrapper function for TRSgep_nakatsukasa which solves the trust region
% subproblem exactly without iterations.
%
% minimize <eta,grad> + .5*<eta,Hess(eta)>
% subject to <eta,eta> <= Delta^2
%
% See also: TRSgep_nakatsukasa, tCG, trustregions

% This file is part of Manopt: www.manopt.org.
% Original author: Victor Liao, June 13, 2022.
% Contributors: 
% Change log: 
    
    M = problem.M;

    numit = 1; % Only meaningful for tCG
    stop_inner = 7; % Only meaningful for tCG
    
    B = tangentorthobasis(M, x);
    H = hessianmatrix(problem, x, B);
    
    n = size(H,1);

    grad_vec = tangent2vec(M, x, B, grad);

    [eta_vec, ~] = TRSgep_nakatsukasa(H, grad_vec, eye(n), Delta);
    eta = lincomb(M, x, B, eta_vec);
    
    Heta_vec = H*eta_vec;
    Heta = lincomb(M, x, B, Heta_vec);

%   Test optimality
%
  assert(all(abs(Heta_vec - tangent2vec(M, x, B, Heta)) <= 1e-10));
  assert(norm(eta_vec)<= Delta + 1e-10, 'cond1');
  lambda_star = (-grad_vec(1) - Heta_vec(1))/ eta_vec(1);
  assert(lambda_star >= 0 || abs(lambda_star) <= 1e-10, 'cond2');
  assert(all(abs(lambda_star * eta_vec + grad_vec + Heta_vec) <= 1e-10), 'cond3');
  assert(abs(lambda_star * (Delta -norm(eta_vec))) <=1e-10, 'cond4');
  assert(all(eig(H + lambda_star * eye(n)) > 0), 'not PD');

