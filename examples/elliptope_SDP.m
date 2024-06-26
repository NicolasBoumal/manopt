function [Y, problem, S] = elliptope_SDP(A, p, Y0)
% Solver for semidefinite programs (SDPs) with unit diagonal constraints.
% 
% function [Y, problem, S] = elliptope_SDP(A)
% function [Y, problem, S] = elliptope_SDP(A, p)
% function [Y, problem, S] = elliptope_SDP(A, p, Y0)
%
% A is a real, symmetric matrix of size n.
%
% This function uses a local optimization method in Manopt to solve the SDP
%
%   min_X  trace(A*X)  s.t.  diag(X) = 1 and X is positive semidefinite.
%
% The symmetric matrix X of size n is parameterized as X = Y*Y',
% where Y has size n x p. By default, p is taken large enough
% (about sqrt(2n)) to ensure that there exists an optimal X whose rank is
% smaller than p. Then, the SDP is equivalent to the new problem in Y:
%
%   min_Y  trace(Y'*A*Y)  s.t.  diag(Y*Y') = 1.
%
% The constraints on Y require each row of Y to have unit norm, which is
% why Manopt is appropriate software to solve this problem. An optional
% initial guess can be specified via the input Y0.
%
% See the paper below for theory, including a proof that, for almost all A,
% second-order critical points of the problem in Y are globally optimal.
% In other words: there are no local traps in Y, despite non-convexity.
%
% Outputs:
%
%       Y: is the best point found (an nxp matrix with unit norm rows.)
%          To find X, form Y*Y' (or, more efficiently, study X through Y.)
% 
%       problem: is the Manopt problem structure used to produce Y.
% 
%       S: is a dual optimality certificate (a symmetric matrix of size n,
%          sparse if A is sparse). The optimality gap (in the cost
%          function) is at most n*min(eig(S)), for both Y and X = Y*Y'.
%          Hence, if min(eig(S)) is close to zero, Y is close to globally
%          optimal. This can be computed via eigs(S, 1, 'SR').
% 
% Paper: https://arxiv.org/abs/1606.04970
%
% @inproceedings{boumal2016bmapproach,
%   author  = {Boumal, N. and Voroninski, V. and Bandeira, A.S.},
%   title   = {The non-convex {B}urer-{M}onteiro approach works on smooth semidefinite programs},
%   booktitle={Neural Information Processing Systems (NIPS 2016)},
%   year    = {2016}
% }
% 
% See also: maxcut elliptope_SDP_complex manoptlift burermonteirolift

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, June 28, 2016
% Contributors:
% Change log:
%
%    Xiaowen Jiang Aug. 20, 2021
%       Added code to showcase how AD would compute egrad and ehess.
%
%    NB, June 26, 2024
%       Revamped the example for a cleaner flow.

    % If no inputs are provided, since this is an example file, generate
    % a random sparse cost matrix. This is for illustration purposes only.
    if ~exist('A', 'var') || isempty(A)
        n = 100;
        A = abs(sprandsym(n, .1));
    end

    n = size(A, 1);
    assert(n >= 2, 'A must be at least 2x2.');
    assert(isreal(A), 'A must be real.');
    assert(size(A, 2) == n, 'A must be square.');
    
    % Force A to be symmetric
    A = (A+A.')/2;
    
    % By default, pick a sufficiently large p (number of columns of Y).
    if ~exist('p', 'var') || isempty(p)
        p = ceil(sqrt(8*n+1)/2);
    end
    
    assert(p >= 2 && p == round(p), 'p must be an integer >= 2.');

    % Pick the manifold of n-by-p matrices with unit-norm rows.
    manifold = obliquefactory(n, p, 'rows');
    
    problem.M = manifold;
    
    
    % These three, quick commented lines of code are sufficient to define
    % the cost function and its derivatives. This is good code to write
    % when prototyping. Below, a more advanced use of Manopt is shown,
    % where the redundant computation A*Y is avoided between the gradient
    % and the cost evaluation.
    % % problem.cost  = @(Y) .5*sum((A*Y).*Y, 'all');
    % % problem.egrad = @(Y) A*Y;
    % % problem.ehess = @(Y, Ydot) A*Ydot;

    
    % Products with A dominate the cost, hence we store the result.
    % This allows to share the results among cost, grad and hess.
    % This is completely optional.
    function store = prepare(Y, store)
        if ~isfield(store, 'AY')
            store.AY = A*Y;
        end
    end
    
    % Define the cost function to be /minimized/.
    problem.cost = @cost;
    function [f, store] = cost(Y, store)
        store = prepare(Y, store);
        f = .5*sum(store.AY .* Y, 'all');
    end

    % Define the Euclidean gradient.
    problem.egrad = @egrad;
    function [G, store] = egrad(Y, store)
        store = prepare(Y, store);
        G = store.AY;
    end

    % If you want to, you can specify the Euclidean Hessian as well.
    problem.ehess = @ehess;
    function [H, store] = ehess(Y, Ydot, store) %#ok<INUSD>
        H = A*Ydot;
    end


    % An alternative way to compute the gradient and the hessian is to use 
    % automatic differentiation via the tool manoptAD.
    %
    % Simply define the cost function without the store structure, then
    % call the tool -- this requires the Deep Learning toolbox.
    %
    % problem.cost = @(Y) .5*sum((A*Y) .* Y, 'all');
    % problem = manoptAD(problem);


    % If no initial guess is available, tell Manopt to use a random one.
    if ~exist('Y0', 'var') || isempty(Y0)
        Y0 = [];
    end

    % Call your favorite solver.
    opts = struct();
    opts.verbosity = 2;      % Set to 0 for no output, 2 for normal output
    opts.maxinner = 500;     % maximum Hessian calls per iteration
    opts.tolgradnorm = 1e-6; % tolerance on gradient norm
    Y = trustregions(problem, Y0, opts);
    
    % If required, produce an optimality certificate.
    if nargout >= 3
        S = A - spdiags(sum((A*Y).*Y, 2), 0, n, n);
    end

end
