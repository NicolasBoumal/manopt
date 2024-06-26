function lift = hadamardlift(set, n, m)
% Hadamard lift Y -> Y.^2 for simplex, stochastic matrices, nonnegative orthant
%
% function lift = hadamardlift('simplex', n)
% function lift = hadamardlift('simplex', n, m)
% function lift = hadamardlift('colstochastic', n, m)
% function lift = hadamardlift('rowstochastic', n, m)
% function lift = hadamardlift('nonnegative', n)
% function lift = hadamardlift('nonnegative', n, m)
%
% This function produces a lift structure to be used with manoptlift.
%
% If m is omitted, then m = 1 by default.
%
% The downstairs manifold consists of real matrices of size n-by-m,
% that is, lift.N = euclideanfactory(n, m).
%
% The upstairs manifold M consists of real matrices Y of size n-by-m,
% possibly restricted to a submanifold selected by the first input string:
%
%   'simplex'       - lift.M = spherefactory(n, m)
%                     That is a unit sphere in R^(n x m)
%   'colstochastic' - lift.M = obliquefactory(n, m, 'cols')
%                     That is a product of m spheres in R^n
%   'rowstochastic' - lift.M = obliquefactory(n, m, 'rows')
%                     That too is a product, of n spheres in R^m
%   'nonnegative'   - lift.M = euclideanfactory(n, m)
%                     That is all of R^(n x m)
%
% The lift is phi(Y) = Y.^2 (entrywise squaring). The image of the lift
% downstairs, that is, phi(M), is determined by the manifold upstairs:
%
%   'simplex'       - matrices of size n-by-m with nonnegative entries
%                     such that the sum of all entries is 1.
%   'colstochastic' - matrices of size n-by-m with nonnegative entries
%                     whose individual columns each sum to 1.
%   'rowstochastic' - matrices of size n-by-m with nonnegative entries
%                     whose individual rows each sum to 1.
%   'nonnegative'   - matrices of size n-by-m with nonnegative entries.
%   
% See https://arxiv.org/abs/2207.03512, Sections 2.1 and 4 for theoretical
% properties of these lifts, e.g., the fact that second-order critical
% points for the problem upstairs map to first-order stationary points for
% the problem downstairs, and also that local minima upstairs map to local
% minima downstairs.
%
% See also: manoptlift burermonteiroLRlift

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, June 19, 2024.
% Contributors: 
% Change log: 

    % TODO: consider adding 'orthostochastic' option, which has M set to
    % the orthogonal matrices. Would need to adapt the general comment
    % about 2=>1 and local=>local.

    if ~exist('m', 'var') || isempty(m)
        m = 1;
    end

    % The space downstairs is R^(n x n)
    lift.N = euclideanfactory(n, m);

    % The space upstairs is determined by the input flag 'set'
    switch lower(set)
        case 'simplex'
            lift.M = spherefactory(n, m);
        case 'rowstochastic'
            lift.M = obliquefactory(n, m, 'rows');
        case 'colstochastic'
            lift.M = obliquefactory(n, m, 'cols');
        case 'nonnegative'
            lift.M = euclideanfactory(n, m);
        otherwise
            error('The set selector string is not recognized.');
    end

    % The lift is the map phi : M -> N which squares individual entries.
    lift.phi = @(Y) Y.*Y;

    % This map is well defined on all of E = R^(n x m), which is the space
    % in which M is embedded.
    % Thus, below we describe the derivatives of phi : E -> N, and set the
    % boolean flag lift.embedded to true so that Manopt knows it needs to
    % adapt them to the specific manifold M.
    lift.embedded = true;

    % Dphi(Y)[V] is the differential of phi : E -> N at Y along V.
    lift.Dphi = @(Y, V) 2*Y.*V;

    % Dphi*(Y)[U] is the adjoint of Dphi(Y) with respect to the usual trace
    % inner products on E and N (both matrix spaces).
    lift.Dphit = @(Y, U) 2*Y.*U;

    % Given W in the Euclidean space downstairs, let h(Y) = <phi(Y), W>,
    % where <., .> is the trace inner product. Then, hesshw computes the
    % Hessian of h : E -> R at Y along V.
    lift.hesshw = @(Y, V, W) 2*V.*W;

end
