function lift = burermonteirolift(constraint, n, p, safety_flag)
% Burer-Monteiro lift Y -> YY' for psd matrices of size n and rank <= p,
% possibly satisfying an additional constraint selected with a flag.
%
% function lift = burermonteirolift('free', n, p)
% function lift = burermonteirolift('unittrace', n, p)
% function lift = burermonteirolift('unitdiag', n, p)
% function lift = burermonteirolift(constraint, n, p, 'symmetric')
%
% This function produces a lift structure to be used with manoptlift.
% 
% The upstairs manifold M consists of real matrices Y of size n-by-p,
% possibly restricted to a submanifold selected by the first input string:
%
%   'free' or '' - lift.M = euclideanfactory(n, p);
%                  No restriction on the factor Y.
%   'unittrace'  - lift.M = spherefactory(n, p);
%                  Y has Frobenius norm 1, so that trace(YY') = 1.
%   'unitdiag'   - lift.M = obliquefactory(p, n, true);
%                  Y has rows of unit norm, so that diag(YY') is all 1.
%
% The downstairs manifold N is the space of real matrices of size n-by-n,
% with the euclideanlargefactory representation to allow efficient use of
% sparsity, rank and other structure.
%
% The lift is phi(Y) = Y*Y' which is usually called the Burer-Monteiro
% lift or factorization or parameterization.
%
% The image of the lift downstairs, that is, phi(M), is determined by the
% manifold upstairs. All of them consist of symmetric positive semidefinite
% matrices of size n and rank at most p, possibly satisfying additional
% constraints:
%
%   'free' or '' - no additional constraint on X
%   'unittrace'  - trace(X) = 1
%   'unitdiag'   - diag(X) is all 1
%
% If the optional 4th input is set to 'symmetric', then the lift assumes
% the downstairs problem used with the lift is actually defined over
% symmetric matrices, so that the Euclidean gradient and Hessian are
% symmetric matrices. This enables some speed ups.
%   
% See https://arxiv.org/abs/2207.03512, Sections 2.2 and 4 for theoretical
% properties of these lifts, e.g., the fact that second-order critical
% points for the problem upstairs map to first-order stationary points for
% the problem downstairs, and also that local minima upstairs map to local
% minima downstairs.
% 
% See also: manoptlift burermonteiroLRlift euclideanlargefactory

    % TODO: add identity block diagonal constraint.
    % TODO: write a complex version.
    % TODO: determine if it would help to have a symmetric version of /
    %       a symmetric format in euclideanlargefactory. Mind Dphi.

    if ~exist('constraint', 'var') || isempty(constraint)
        constraint = '';
    end

    if ~exist('safety_flag', 'var') || isempty(safety_flag) ...
                                    || ~strcmp(safety_flag, 'symmetric')
        guaranteed_symmetry = false;
    else
        guaranteed_symmetry = true;
    end

    % The space downstairs is R^(n x n), with support for large matrices.
    Rnn = euclideanlargefactory(n, n);
    lift.N = Rnn;

    % The space upstairs is determined by the input flag 'constraint'
    switch lower(constraint)
        case {'free', ''}
            lift.M = euclideanfactory(n, p);
        case 'unittrace'
            lift.M = spherefactory(n, p);
        case 'unitdiag'
            lift.M = obliquefactory(p, n, true);
        otherwise
            error('The constraint string is not recognized.');
    end

    % The lift is the map phi : M -> N such that phi(Y) = Y*Y'.
    % This image is expressed in the euclideanlargefactory format, which
    % allows to store a large matrix X as a pair (L, R) (in a structure)
    % such that X = L*R'.
    lift.phi = @(Y) struct('L', Y, 'R', Y);

    % This map is well defined on all of E = R^(n x p), which is the space
    % in which M is embedded.
    % Thus, below we describe the derivatives of phi : E -> N, and set the
    % boolean flag lift.embedded to true so that Manopt knows it needs to
    % adapt them to the specific manifold M.
    lift.embedded = true;
    
    % Dphi(Y)[V] = V*Y' + Y*V' is the differential of phi:E->N at Y along V
    lift.Dphi = @(Y, V) struct('L', [V, Y], 'R', [Y, V]);

    % Dphi*(Y)[U] = (U+U')*Y is the adjoint of Dphi(Y) with respect to the
    % usual trace inner products on E and N (both matrix spaces).
    % Thus, Y lives in E (on M), and U lives in N, that is, Rnn.
    % The output is in E. If the user does not guarantee that U is
    % symmetric, then we must compute U*Y and U'*Y separately.
    if ~guaranteed_symmetry
        lift.Dphit = @(Y, U) Rnn.times(U, Y) + Rnn.transpose_times(U, Y);
    else
        lift.Dphit = @(Y, U) 2*Rnn.times(U, Y);
    end

    % Given W in the Euclidean space downstairs, let h(Y) = <phi(Y), W>,
    % where <., .> is the trace inner product. Then, hesshw computes the
    % Hessian of h : E -> R at Y along V, which is (W+W')*V.
    % If the user does not guarantee W is symmetric, do the safe thing.
    if ~guaranteed_symmetry
        lift.hesshw = @(Y, V, W) Rnn.times(W, V) + ...
                                 Rnn.transpose_times(W, V);
    else
        lift.hesshw = @(Y, V, W) 2*Rnn.times(W, V);
    end

end
