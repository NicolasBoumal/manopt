function M = stiefelfactory(n, p, k)
% Returns a manifold structure to optimize over orthonormal matrices.
%
% function M = stiefelfactory(n, p)
% function M = stiefelfactory(n, p, k)
%
% The Stiefel manifold is the set of orthonormal nxp matrices. If k
% is larger than 1, this is the Cartesian product of the Stiefel manifold
% taken k times. The metric is such that the manifold is a Riemannian
% submanifold of R^nxp equipped with the usual trace inner product, that
% is, it is the usual metric.
%
% Points are represented as matrices X of size n x p x k (or n x p if k=1,
% which is the default) such that each n x p matrix is orthonormal,
% i.e., X'*X = eye(p) if k = 1, or X(:, :, i)' * X(:, :, i) = eye(p) for
% i = 1 : k if k > 1. Tangent vectors are represented as matrices the same
% size as points.
%
% By default, k = 1.
%
% See also: grassmannfactory rotationsfactory

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
%  July  5, 2013 (NB) : Added ehess2rhess.
%  Jan. 27, 2014 (BM) : Bug in ehess2rhess corrected.
%  June 24, 2014 (NB) : Added true exponential map and changed the randvec
%                       function so that it now returns a globally
%                       normalized vector, not a vector where each
%                       component is normalized (this only matters if k>1).
%  July 17, 2018 (NB) : Now both QR (default) and polar retractions are
%                       directly accessible, and their inverses are also
%                       implemented.

    
    if ~exist('k', 'var') || isempty(k)
        k = 1;
    end
    
    if k == 1
        M.name = @() sprintf('Stiefel manifold St(%d, %d)', n, p);
    elseif k > 1
        M.name = @() sprintf('Product Stiefel manifold St(%d, %d)^%d', n, p, k);
    else
        error('k must be an integer no less than 1.');
    end
    
    M.dim = @() k*(n*p - .5*p*(p+1));
    
    M.inner = @(x, d1, d2) d1(:).'*d2(:);
    
    M.norm = @(x, d) norm(d(:));
    
    M.dist = @(x, y) error('stiefel.dist not implemented yet.');
    
    M.typicaldist = @() sqrt(p*k);
    
    M.proj = @projection;
    function Up = projection(X, U)
        
        XtU = multiprod(multitransp(X), U);
        symXtU = multisym(XtU);
        Up = U - multiprod(X, symXtU);
        
% The code above is equivalent to, but much faster than, the code below.
%         
%     Up = zeros(size(U));
%     function A = sym(A), A = .5*(A+A'); end
%     for i = 1 : k
%         Xi = X(:, :, i);
%         Ui = U(:, :, i);
%         Up(:, :, i) = Ui - Xi*sym(Xi'*Ui);
%     end

    end
    
    M.tangent = M.proj;
    
    % For Riemannian submanifolds, converting a Euclidean gradient into a
    % Riemannian gradient amounts to an orthogonal projection.
	M.egrad2rgrad = M.proj;
    
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, H)
        XtG = multiprod(multitransp(X), egrad);
        symXtG = multisym(XtG);
        HsymXtG = multiprod(H, symXtG);
        rhess = projection(X, ehess - HsymXtG);
    end
    
    M.retr_qr = @retraction_qr;
    function Y = retraction_qr(X, U, t)
        if nargin < 3
            Y = X + U;
        else
            Y = X + t*U;
        end
        for kk = 1 : k
            [Q, R] = qr(Y(:, :, kk), 0);
            % The instruction with R ensures we are not flipping signs
            % of some columns, which should never happen in modern Matlab
            % versions but may be an issue with older versions.
            Y(:, :, kk) = Q * diag(sign(sign(diag(R))+.5));
        end
    end

    % This inverse retraction is valid for both the QR retraction and the
    % polar retraction.
    M.invretr_qr = @invretr_qr;
    function U = invretr_qr(X, Y)
        XtY = multiprod(multitransp(X), Y);
        R = zeros(p, p, k);
        for kk = 1 : k
            % For each slice, assuming the inverse retraction is well
            % defined for the given inputs, we have:
            %   X + U = YR
            % Left multiply with X' to get
            %   I + X'U = X'Y M
            % Since X'U is skew symmetric for a tangent vector U at X, add
            % up this equation with its transpose to get:
            %   2I = (X'Y) R + R' (X'Y)'
            % Contrary to the polar factorization, here R is not symmetric
            % but it is upper triangular. As a result, this is not a
            % Sylvester equation and we must solve it differently.
            R(:, :, kk) = solve_for_triu(XtY(:, :, kk), 2*eye(p));
            % Then,
            %   U = YR - X
            % which is what we compute below.
        end
        U = multiprod(Y, R) - X;
    end
    
    M.retr_polar = @retraction_polar;
    function Y = retraction_polar(X, U, t)
        if nargin < 3
            Y = X + U;
        else
            Y = X + t*U;
        end
        for kk = 1 : k
            [u, s, v] = svd(Y(:, :, kk), 'econ'); %#ok
            Y(:, :, kk) = u*v';
        end
    end
    
    % This inverse retraction is valid for both the QR retraction and the
    % polar retraction.
    M.invretr_polar = @invretr_polar;
    function U = invretr_polar(X, Y)
        XtY = multiprod(multitransp(X), Y);
        MM = zeros(p, p, k);
        for kk = 1 : k
            % For each slice, assuming the inverse retraction is well
            % defined for the given inputs, we have:
            %   X + U = YM
            % Left multiply with X' to get
            %   I + X'U = X'Y M
            % Since X'U is skew symmetric for a tangent vector U at X, add
            % up this equation with its transpose to get:
            %   2I = (X'Y) M + M' (X'Y)'
            %      = (X'Y) M + M (X'Y)'   since M is symmetric.
            % Solve for M symmetric with a call to sylvester:
            MM(:, :, kk) = sylvester(XtY(:, :, kk), XtY(:, :, kk)', 2*eye(p));
            % Then,
            %   U = YM - X
            % which is what we compute below.
        end
        U = multiprod(Y, MM) - X;
    end
    
    % By default, we use the QR retraction
    M.retr = M.retr_qr;
    M.invretr = M.invretr_qr;

    M.exp = @exponential;
    function Y = exponential(X, U, t)
        if nargin == 2
            t = 1;
        end
        tU = t*U;
        Y = zeros(size(X));
        for i = 1 : k
            % From a formula by Ross Lippert, Example 5.4.2 in AMS08.
            Xi = X(:, :, i);
            Ui = tU(:, :, i);
            Y(:, :, i) = [Xi Ui] * ...
                         expm([Xi'*Ui , -Ui'*Ui ; eye(p) , Xi'*Ui]) * ...
                         [ expm(-Xi'*Ui) ; zeros(p) ];
        end
        
    end

    M.hash = @(X) ['z' hashmd5(X(:))];
    
    M.rand = @random;
    function X = random()
        X = zeros(n, p, k);
        for i = 1 : k
            [Q, unused] = qr(randn(n, p), 0);  %#ok<ASGLU>
            X(:, :, i) = Q;
        end
    end
    
    M.randvec = @randomvec;
    function U = randomvec(X)
        U = projection(X, randn(n, p, k));
        U = U / norm(U(:));
    end
    
    M.lincomb = @matrixlincomb;
    
    M.zerovec = @(x) zeros(n, p, k);
    
    M.transp = @(x1, x2, d) projection(x2, d);
    
    M.vec = @(x, u_mat) u_mat(:);
    M.mat = @(x, u_vec) reshape(u_vec, [n, p, k]);
    M.vecmatareisometries = @() true;

end


% Support function for the QR inverse retraction
function X = solve_for_triu(A, H)
    % Given A of size pxp and H (symmetric) of size pxp,
    % Solves the linear equation AX + X'A' = H for X upper triangular.
    % The total cost is O(p^4). Perhaps this can be improved? With
    % preprocessing of A? LU?
    p = size(A, 1);
    X = zeros(p, p);
    for pp = 1 : p
        b = H(1:pp, pp);
        b(end) = b(end)/2;
        b(1:end-1) = b(1:end-1) - X(1:pp-1, 1:pp-1)'*A(pp, 1:pp-1)';
        X(1:pp, pp) = A(1:pp, 1:pp) \ b;
    end
end
