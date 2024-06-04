function desingularization_matrix_completion()
% Example file to test the manifold desingularizationfactory.
%
% See also: low_rank_matrix_completion

% This file is part of Manopt: www.manopt.org.
% Original author: Quentin Rebjock and Nicolas Boumal, May 2024.
% Contributors: 
% Change log: 
    
    % Random data generation. First, choose the size of the problem.
    % We will complete a matrix of size mxn of rank <= r.
    m = 200;
    n = 500;
    r = 5;
    alpha = rand(); % parameter for the Riemannian metric
    
    % Generate a random mxn matrix A of rank r
    L = randn(m, r);
    R = randn(n, r);
    A = L*R';
    
    % Generate a random mask for observed entries: P(i, j) = 1 if the entry
    % (i, j) of A is observed, and 0 otherwise.
    fraction = 4*(m + n - r)*r/(m*n);    % oversampling factor
    P = sparse(rand(m, n) <= fraction);
    
    % Data: we know the nonzero entries in P.*A:
    PA = P.*A;
    
    problem.M = desingularizationfactory(m, n, r, alpha);
    
    % Define the problem cost function. The input X is a structure with
    % fields U, S, V representing a matrix of rank <= r as U*S*V'.
    % f(X) = 1/2 * || P.*(X-A) ||^2
    problem.cost = @cost;
    function f = cost(X)
        % Note that it is very much inefficient to explicitly construct the
        % matrix X in this way. Seen as we only need to know the entries
        % of Xmat corresponding to the mask P, it would be far more
        % efficient to compute only those. We keep the code simple here.
        Xmat = X.U*X.S*X.V';
        f = .5*norm(P.*Xmat - PA , 'fro')^2;
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
    
    % The Hessian is optional, but it's nice if you have it.
    % Define the Euclidean Hessian of the cost at X, along H, where H is
    % represented as a tangent vector: a structure with fields K and Vp.
    % The output is represented as a [TODO].
    % This is the directional derivative of nabla f(X) at X along Xdot:
    % nabla^2 f(X)[Xdot] = P.*Xdot.
    problem.ehess = @ehess;
    function ehess = ehess(X, H)
        % Same comment here about explicitly constructing the ambient
        % vector as an mxn matrix Xdot: we only need its entries
        % corresponding to the mask P, and this could be computed
        % efficiently. The following:
        Xdot = H.K*X.V' + X.U*X.S*H.Vp';
        % is equivalent to:
        % Xdot = M.tangent2ambient(X, H).Y; % TODO: replace .Y with .Xdot?
        ehess = P.*Xdot;
    end
    
    X = trustregions(problem);

    fprintf('Total error: %g\n', ...
            norm(problem.M.triplet2matrix(X) - A, 'fro'));

end
