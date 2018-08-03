function Test_fixedrankembeddedfactory()
% Given partial observation of a low rank matrix, attempts to complete it.
%
% function low_rank_matrix_completion()
%
% This example demonstrates how to use the geometry factory for the
% embedded submanifold of fixed-rank matrices, fixedrankembeddedfactory.
% This geometry is described in the paper
% "Low-rank matrix completion by Riemannian optimization"
% Bart Vandereycken - SIAM Journal on Optimization, 2013.
%
% This can be a starting point for many optimization problems of the form
%
% minimize f(X) such that rank(X) = k, size(X) = [m, n].
%
% Input:  None. This example file generate random data.
% 
% Output: None.

% This file is part of Manopt and is copyrighted. See the license file.
% 
% Main author: Nicolas Boumal, July 15, 2014
% Contributors:
% 
% Change log:
% 
    
    % Choose the size of the problem:
    % We will complete a matrix of size mxn of rank k.
    m = 2000;
    n = 5000;
    k = 10;
    % Generate a random mxn matrix A of rank k
    L = randn(m, k);
    R = randn(n, k);
    A = L*R';
    % Generate a random mask for observed entries: P(i, j) = 1 if the entry
    % (i, j) of A is observed, and 0 otherwise.
    fraction = 4 * k*(m+n-k)/(m*n);
    P = sparse(rand(m, n) <= fraction);
    % Hence, we know the nonzero entries in PA:
    PA = P.*A;
    
    % Pick the manifold of matrices of size mxn of fixed rank k.
    problem.M = fixedrankembeddedfactory(m, n, k);

    % Define the problem cost function. The input X is a structure with
    % fields U, S, V representing a rank k matrix as U*S*V'.
    % f(X) = 1/2 * || P.*(X-A) ||^2
    problem.cost = @cost;
    function f = cost(X)
        % Note that it is very much inefficient to explicitly construct the
        % matrix X in this way. Seen as we only need to know the entries
        % of Xmat corresponding to the mask P, it would be far more
        % efficient to compute those only.
        Xmat = X.U*X.S*X.V';
        f = .5*norm( P.*Xmat - PA , 'fro')^2;
    end

    % Define the Euclidean gradient of the cost function, that is, the
    % gradient of f(X) seen as a standard function of X.
    % nabla f(X) = P.*(X-A)
    problem.egrad = @egrad;
    function G = egrad(X)
        % Same comment here about Xmat.
        Xmat = X.U*X.S*X.V';
        G = P.*Xmat - PA;
    end

    % Define the Euclidean Hessian of the cost at X, along H, where H is
    % represented as a tangent vector.
    % nabla^2 f(X)[Xdot] = P.*Xdot
    % (This is the directional derivative of nabla f(X) at X along Xdot.)
    problem.ehess = @euclidean_hessian;
    function ehess = euclidean_hessian(X, H)
        % The function tangent2ambient transforms H (a tangent vector) into
        % its equivalent ambient vector representation. The output is a
        % structure with fields U, S, V such that U*S*V' is an mxn matrix
        % corresponding to the tangent vector H. Note that there are no
        % additional guarantees about U, S and V. In particular, U and V
        % are not orthonormal.
        ambient_H = problem.M.tangent2ambient(X, H);
        Xdot = ambient_H.U*ambient_H.S*ambient_H.V';
        % Same comment here about explicitly constructing the ambient
        % vector as an mxn matrix Xdot: we only need its entries
        % corresponding to the mask P, and this could be computed
        % efficiently.
        ehess = P.*Xdot;
    end
    

    % Check consistency of the gradient and the Hessian. Useful if you
    % adapt this example for a now cost function and you would like to make
    % sure there is no mistake.
    % warning('off', 'manopt:fixedrankembeddedfactory:exp');
    % checkgradient(problem); pause;
    % checkhessian(problem); pause;
    
    % Compute an initial guess. Points on the manifold are represented as
    % structures with three fields: U, S and V. U and V need to be
    % orthonormal, S needs to be diagonal.
    [U, S, V] = svds(PA, k);
    X0.U = U;
    X0.S = S;
    X0.V = V;
    
    % Minimize the cost function using Riemannian trust-regions, starting
    % from the initial guess X0.
    options.stopfun = @stopifclosedfigure;
    X = trustregions(problem, X0, options);
    
    % The reconstructed matrix is X, represented as a structure with fields
    % U, S and V.
    Xmat = X.U*X.S*X.V';
    fprintf('||X-A||_F = %g\n', norm(Xmat - A, 'fro'));

    % If the problem has a small enough dimension, we may (for analysis
    % purposes) compute the spectrum of the Hessian at a point X. This may
    % help in studying the conditioning of a problem.
    if problem.M.dim() < 100
        fprintf('Computing the spectrum of the Hessian...');
        s = hessianspectrum(problem, X);
        hist(s);
    end
    
end
