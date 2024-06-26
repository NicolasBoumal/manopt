function [x, cutvalue, cutvalue_upperbound, Y] = maxcut(L, r)
% Algorithm to (try to) compute a maximum cut of a graph, via SDP approach.
% 
% function x = maxcut(L)
% function [x, cutvalue, cutvalue_upperbound, Y] = maxcut(L, r)
%
%   This is an example file meant to illustrate functionalities of Manopt.
%   It is not meant to be an efficient implementation of a Max-Cut solver.
%
%   Please see the following repository for more efficient Manopt code:
%   https://github.com/NicolasBoumal/maxcut
%
%
% L is the Laplacian matrix describing the graph to cut. The Laplacian of a
% graph is L = D - A, where D is the diagonal degree matrix (D(i, i) is the
% sum of the weights of the edges adjacent to node i) and A is the
% symmetric adjacency matrix of the graph (A(i, j) = A(j, i) is the weight
% of the edge joining nodes i and j). If L is sparse, this will be
% exploited.
%
% If the graph has n nodes, then L is nxn and the output x is a vector of
% length n such that x(i) is +1 or -1. This partitions the nodes of the
% graph in two classes, in an attempt to maximize the sum of the weights of
% the edges that go from one class to the other (MAX CUT problem).
%
% cutvalue is the sum of the weights of the edges 'cut' by the partition x.
%
% If the algorithm reached the global optimum of the underlying SDP
% problem, then it produces an upperbound on the maximum cut value. This
% value is returned in cutvalue_upperbound if it is found. Otherwise, that
% output is set to NaN.
%
% If r is specified (by default, r = n), the algorithm stops at rank r.
% This may prevent the algorithm from reaching a globally optimal solution
% for the underlying SDP problem, but can greatly help in keeping the
% execution time under control. If a global optimum of the SDP is reached
% before rank r, the algorithm stops then of course.
%
% Y is a matrix of size nxp, with p <= r, such that X = Y*Y' is the best
% solution found for the underlying SDP problem. If cutvalue_upperbound is
% not NaN, then Y*Y' is optimal for the SDP and cutvalue_upperbound is its
% cut value.
% 
% By Goemans and Williamson 1995, it is known that if the optimal value of
% the SDP is reached, then the returned cut, in expectation, is at most at
% a fraction 0.878 of the optimal cut. (This is not exactly valid because
% we do not use random projection here; sign(Y*randn(size(Y, 2), 1)) gives
% a cut that respects this statement -- it's usually worse though).
%
% The algorithm is essentially that of:
% 
% Journee, Bach, Absil and Sepulchre, SIAM 2010
% Low-rank optimization on the cone of positive semidefinite matrices.
%
% but on the simpler oblique geometry (product of spheres), to avoid theory
% breakdown at rank deficient Y.
%
% It is itself based on the famous SDP relaxation of MAX CUT:
% Goemans and Williamson, 1995
% Improved approximation algorithms for maximum cut and satisfiability
% problems using semidefinite programming
%
% and the related work of Burer and Monteiro 2003, 2005.
% 
% See also: elliptope_SDP burermonteirolift

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, July 18, 2013
% Contributors:
% Change log:
%   
%   April 3, 2015 (NB):
%       L products now counted with the new shared memory system. This is
%       more reliable and more flexible than using a global variable.
%   Aug  20, 2021 (XJ):
%       Added AD to compute the egrad and the ehess
%   March 25, 2024 (NB):
%       Using obliquefactory rather than elliptopefactory by default as
%       this takes better care of rank deficient matrices.
%   June 26, 2024 (NB):
%       Revamped the example to be more in line with how Manopt evolved.


    % If no inputs are provided, generate a random graph Laplacian.
    % This is for illustration purposes only.
    if ~exist('L', 'var') || isempty(L)
        n = 100;
        A = abs(sprandsym(n, .1)); % random sparse adjacency matrix
        D = diag(sum(A, 2));
        L = D-A;
    end


    n = size(L, 1);
    assert(size(L, 2) == n, 'L must be square.');

    if ~exist('r', 'var') || isempty(r) || r > n
        r = n;
    end
    
    % We let the rank increase bit by bit. Each rank value generates a cut.
    % We have to go up in the rank to eventually find a certificate of SDP
    % optimality. This in turn provides an upperbound on the MAX CUT value
    % and ensure that we're doing well, according to Goemans and
    % Williamson's argument. In practice though, the good cuts often come
    % up for low rank values, so we better keep track of the best one.
    best_x = ones(n, 1);
    best_cutvalue = 0;
    cutvalue_upperbound = NaN;
    
    time = [];
    cost = [];
    
    for rr = 2 : r

        manifold = obliquefactory(n, rr, 'rows');
        
        if rr == 2
            
            % At first, for rank 2, generate a random point.
            Y0 = manifold.rand();
             
        else
            
            % To increase the size, we could just add a column of zeros to
            % the Y matrix. Unfortunately, this lands us in a saddle point.
            % To escape from the saddle, we may compute an eigenvector of
            % Sy associated to a negative eigenvalue: that yields a
            % (second order) descent direction Z.
            % Sy is linked to dual certificates for the SDP (Journée et al)
            % A more pragmatic approach is to simply add a random column
            % to Y with small norm, and renormalize.
            Y0 = [Y, zeros(n, 1)];
            LY0 = L*Y0;
            Dy = spdiags(sum(LY0.*Y0, 2), 0, n, n);
            Sy = (Dy - L)/4;
            % Find the smallest (the "most negative") eigenvalue of Sy.
            [v, s] = eigs(Sy, 1, 'smallestreal');
            % If there is no negative eigenvalue for Sy, than we are not at
            % a saddle point: we're actually done!
            if s >= -1e-8
                % We can stop here: we found the global optimum of the SDP,
                % and hence the reached cost is a valid upper bound on the
                % maximum cut value.
                cutvalue_upperbound = max(-[info.cost]);
                break;
            end
            
            % This is our escape direction.
            Z = manifold.proj(Y0, [zeros(n, rr-1) v]);
            
            % Now make a step in the Z direction to escape from the saddle.
            % This is merely a heuristic: it may be better to us a
            % line-search on the stepsize to guarantee cost decrease.
            stepsize = 1;
            Y0 = manifold.retr(Y0, Z, stepsize);
            
        end
        
        % Use the Riemannian optimization algorithm lower in this file to
        % reach a critical point (typically a local optimizer) of the
        % max-cut cost with rank at most rr, starting from Y0.
        [Y, info] = maxcut_boundedrank(L, Y0);
        
        % Some info logging.
        thistime = [info.time];
        if ~isempty(time)
            thistime = time(end) + thistime;
        end
        time = [time, thistime]; %#ok<AGROW>
        cost = [cost, [info.cost]]; %#ok<AGROW>

        % Time to turn the matrix Y into a cut.
        % We can either do the random rounding as follows:
        %
        % x = sign(Y*randn(rr, 1));
        %
        % or extract the "PCA direction" of the points in Y and cut
        % orthogonally to that direction, as follows (seems faster than
        % calling svds):
        [U, ~, ~] = svd(Y, 0);
        u = U(:, 1);
        x = sign(u);

        cutvalue = (x'*L*x)/4;
        if cutvalue > best_cutvalue
            best_x = x;
            best_cutvalue = cutvalue;
        end
        
    end
    
    x = best_x;
    cutvalue = best_cutvalue;
    
    clf;
    plot(time, -cost, '.-');
    xlabel('Time [s]');
    ylabel('Relaxed cut value');
    title(sprintf('Max-Cut value upper bound: %g. Best cut found: %g.', ...
            cutvalue_upperbound, cutvalue));

end


function [Y, info] = maxcut_boundedrank(L, Y)
% Try to solve the rank-r relaxed max-cut program, based on the
% Laplacian of the graph L and an initial guess Y. L is nxn and Y is nxr.

    [n, r] = size(Y);
    assert(all(size(L) == n));
   
    % Constrain the rows of Y (of size nxr) to have unit norm.
    manifold = obliquefactory(n, r, 'rows');
    
    problem.M = manifold;
    
    % For rapid prototyping, the next three lines suffice to describe the
    % cost function and its gradient and Hessian (here expressed using the
    % Euclidean gradient and Hessian).
    %
    % problem.cost  = @(Y) -sum(Y.*(L*Y), 'all')/4;   % = -trace(Y.'*LY)/4;
    % problem.egrad = @(Y) -(L*Y)/2;
    % problem.ehess = @(Y, U) -(L*U)/2;

    % It's also possible to use automatic differentiation (AD) instead of
    % implementing the gradient and Hessian by hand, though that is slower.
    % 
    % problem.cost  = @(Y) -sum(Y.*(L*Y), 'all')/4;
    % problem = manoptAD(problem);
    
    % Instead of the prototyping version, the functions below describe the
    % cost, gradient and Hessian using the caching system (the store
    % structure). This makes it possible to avoid redundant computations of
    % products with the matrix L, which are likely to be the expensive bit.
    % For analysis, these multiplications are counted with Manopt counters.

    % For every visited point Y, we need L*Y. This function makes sure
    % the quantity L*Y is available, but only computes it if it wasn't
    % already computed.
    function store = prepare(Y, store)
        if ~isfield(store, 'LY')
            % Compute and store the product for the current point Y.
            store.LY = L*Y;
            store = incrementcounter(store, 'Lproducts', size(Y, 2));
        end
    end

    problem.cost = @cost;
    function [f, store] = cost(Y, store)
        store = prepare(Y, store);
        LY = store.LY;
        f = -(Y(:)'*LY(:))/4;   % = -trace(Y'*LY)/4; but faster
    end

    problem.egrad = @egrad;
    function [g, store] = egrad(Y, store)
        store = prepare(Y, store);
        LY = store.LY;
        g = -LY/2;
    end

    problem.ehess = @ehess;
    function [h, store] = ehess(Y, U, store)
        store = prepare(Y, store);   % this line is not strictly necessary
        LU = L*U;
        store = incrementcounter(store, 'Lproducts', size(U, 2));
        h = -LU/2;
    end

    % Register a counter to keep track of products with the L matrix.
    stats = statscounters('Lproducts');
    options.statsfun = statsfunhelper(stats);
    
    options.verbosity = 1;

    [Y, Ycost, info] = trustregions(problem, Y, options); %#ok<ASGLU>
    
    fprintf('At rank <= %d, matrix-vector products with L: %d\n\n', ...
             r, max([info.Lproducts]));

end
