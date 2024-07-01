function lift = burermonteiroLRlift(m, n, r)
% Burer-Monteiro lift (L, R) -> L*R.' for m-by-n matrices of rank <= r.
%
% function lift = burermonteiroLRlift(m, n, r)
% function lift = burermonteiroLRlift(m, n, r, lambda)
%
% This function produces a lift structure to be used with manoptlift.
% 
% The upstairs manifold M consists of pairs (L, R) with L of size m-by-r
% and R of size n-by-r, both real.
%
% The downstairs manifold N is the space of real matrices of size m-by-n,
% with the euclideanlargefactory representation to allow efficient use of
% sparsity, rank and other structure.
%
% The lift is phi(L, R) = L*R.' and its image is the set of real matrices
% of size m-by-n with rank at most r.
%   
% See https://arxiv.org/abs/2207.03512, Sections 2.3 and 5 for theoretical
% properties of this lift, e.g., the fact that second-order critical points
% for the problem upstairs map to first-order stationary points for the
% problem downstairs.
%
% By default, lambda = 0. If a nonzero value is provided, then the cost
% function upstairs is regularized as follows:
%
%    g(L, R) = f(LR') + lambda*rho(L, R)
%
% where rho(L, R) = .5*(||L||^2 + ||R||^2) (in Frobenius norms).
% Minimizing g upstairs amounts to minimizing the following downstairs:
%
%    X -> f(X) + lambda*nuclear_norm(X)  s.t.  rank(X) <= r.
%
% Thus, there is a hard-cap on rank, and a low-rank regularizer on top.
% See "Maximum-Margin Matrix Factorization" by Srebro, Rennie and Jaakkola.
% 
% See also: manoptlift burermonteirolift desingularizationfactory
%           fixedrankembeddedfactory euclideanlargefactory

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, June 18, 2024.
% Contributors: 
% Change log: 
%   July 1, 2024 (NB)
%       Added the regularizer rho(L, R) = .5*(||L||^2 + ||R||^2).

    % TODO: complex version

    % Upstairs
    elems.L = euclideanfactory(m, r);
    elems.R = euclideanfactory(n, r);
    LRspace = productmanifold(elems);
    lift.M = LRspace;

    % Downstairs
    Rmn = euclideanlargefactory(m, n);
    lift.N = Rmn;

    % The upstairs space is a linear space, and it is its own embedding
    % space. Thus, the flag lift.embedded can be set equivalently to true
    % or false. The choice induces some effects in the way manoptlift
    % creates the downstairs problem (grad/hess vs egrad/ehess), which in
    % turn has a small (essentially negligible) effect on performance.
    % Based on limited testing, it appears that setting the flag to false
    % is ever so slightly more efficient.
    % If it becomes useful to replace the upstairs space with a submanifold
    % (e.g., restricting L to stiefelfactory), then it would be necessary
    % to set the flag to true.
    lift.embedded = false;

    % phi : M -> N : (L, R) -> L*R'
    % M is the set of pairs (L, R) as structures with fields L and R.
    % N is the set of matrices X of size m-by-n with any of several
    % accepted numerical representations. One of them is to represent X as
    % a structure with fields L and R to mean L*R.', so that there is
    % nothing left to do.
    lift.phi = @(LR) LR;

    % Dphi(L, R)[Ldot, Rdot] = Ldot*R' + L*Rdot' is the differential of phi
    lift.Dphi = @(LR, LRdot) ...
        struct('L', [LRdot.L, LR.L], ...
               'R', [LR.R, LRdot.R]);

    % Dphit(L, R) is the adjoint of Dphi(L, R) with respect to the usual
    % trace inner product over matrices, so that
    % Dphit(L, R)[Xdot] = (Xdot*R, Xdot'*L), as a tangent vector to M.
    lift.Dphit = @(LR, Xdot) ...
        struct('L', Rmn.times(Xdot, LR.R), ...
               'R', Rmn.transpose_times(Xdot, LR.L));

    % Given a matrix W of size m-by-n, let h(L, R) = <phi(L, R), W>,
    % where <., .> denotes the trace inner product.
    % Then, hesshw is the Hessian of h at (L, R) along (Ldot, Rdot),
    % that is, (W*Rdot, W'*Ldot).
    lift.hesshw = @(LR, LRdot, W) ...
        struct('L', Rmn.times(W, LRdot.R), ...
               'R', Rmn.transpose_times(W, LRdot.L));

    % Regularizer rho(L, R) = .5*(||L||^2 + ||R||^2)
    lift.rho = @(LR) .5*(norm(LR.L, 'fro')^2 + norm(LR.R, 'fro')^2);
    lift.gradrho = @(LR) LR;
    lift.hessrho = @(LR, LRdot) LRdot;

end
