function M = rotationsfactory(n, k)
% Returns a manifold structure to optimize over rotation matrices.
% 
% function M = rotationsfactory(n)
% function M = rotationsfactory(n, k)
%
% Special orthogonal group (the manifold of rotations): deals with matrices
% R of size n x n x k (or n x n if k = 1, which is the default) such that
% each n x n matrix is orthogonal, i.e., X'*X = eye(n) if k = 1, or
% X(:, :, i)' * X(:, :, i) = eye(n) for i = 1 : k if k > 1. Furthermore,
% all these matrices have determinant +1.
%
% This is a description of SO(n)^k with the induced metric from the
% embedding space (R^nxn)^k, i.e., this manifold is a Riemannian
% submanifold of (R^nxn)^k endowed with the usual trace inner product.
%
% This is important:
% Tangent vectors are represented in the Lie algebra, i.e., as skew
% symmetric matrices. Use the function M.tangent2ambient(X, H) to switch
% from the Lie algebra representation to the embedding space
% representation. This is often necessary when defining
% problem.ehess(X, H), as the input H will then be a skew-symmetric matrix
% (but the output must not be, as the output is the Hessian in the
% embedding Euclidean space.)
%
% By default, the retraction is only a first-order approximation of the
% exponential. To force the use of a second-order approximation, call
%
%     M.retr = M.retr_polar;
%     M.invretr = M.invretr_polar;
%
% after creating M. This switches from a QR-based computation to an
% SVD-based computation.
%
% For n = 3, the code uses Rodrigues formulas for exp and log. Since most
% optimization algorithms use retractions by default, you can force those
% algorithms to use the Rodrigues formulas as follows:
%  
%     M.retr = M.exp;
%     M.invretr = M.log;
%
%
% By default, k = 1.
%
%
% See also: stiefelfactory

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: Spencer Kraisler
% Change log:
%   Jan. 31, 2013 (NB)
%       Added egrad2rgrad and ehess2rhess
%   Oct. 21, 2016 (NB)
%       Added M.retr2: a second-order retraction based on SVD.
%   July 18, 2018 (NB)
%       Fixed a bug in M.retr2 (only relevant for k > 1).
%       Added inverse retraction as M.invretr.
%       Retraction and inverse also available as M.retr_qr, M.invretr_qr.
%       Renamed M.retr2 to M.retr_polar, and implemented M.invretr_polar.
%       For backward compatibility, M.retr2 is still defined and is now
%       equal to M.retr_polar. There is no M.invretr2.
%   Sep.  06, 2018 (NB)
%       Added M.isotransp, which is equal to M.transp since it is
%       isometric.
%   June 18, 2019 (NB)
%       Using qr_unique for the QR-based retraction.
%   Nov. 19, 2019 (NB)
%       Clarified that the isometric transport is not parallel transport
%       along geodesics.
%   June 10, 2022 (NB)
%       Following input from Spencer Kraisler on Manopt forum, added code
%       for Rodrigues formulas, now used instead of expm / logm for n = 3.

    
    if ~exist('k', 'var') || isempty(k)
        k = 1;
    end
    
    if k == 1
        M.name = @() sprintf('Rotations manifold SO(%d)', n);
    elseif k > 1
        M.name = @() sprintf('Product rotations manifold SO(%d)^%d', n, k);
    else
        error('k must be an integer no less than 1.');
    end
    
    M.dim = @() k*nchoosek(n, 2);
    
    M.inner = @(x, d1, d2) d1(:).'*d2(:);
    
    M.norm = @(x, d) norm(d(:));
    
    M.typicaldist = @() pi*sqrt(n*k);
    
    M.proj = @(X, H) multiskew(multiprod(multitransp(X), H));
    
    M.tangent = @(X, H) multiskew(H);
    
    M.tangent2ambient_is_identity = false;
    M.tangent2ambient = @(X, U) multiprod(X, U);
    
    M.egrad2rgrad = M.proj;
    
    M.ehess2rhess = @ehess2rhess;
    function Rhess = ehess2rhess(X, Egrad, Ehess, H)
        % Reminder : H contains skew-symmeric matrices. The actual
        % direction that the point X is moved along is X*H.
        Xt = multitransp(X);
        XtEgrad = multiprod(Xt, Egrad);
        symXtEgrad = multisym(XtEgrad);
        XtEhess = multiprod(Xt, Ehess);
        Rhess = multiskew( XtEhess - multiprod(H, symXtEgrad) );
    end
    
    % This QR-based retraction is only a first-order approximation
    % of the exponential map, not a second-order one.
    M.retr_qr = @retraction_qr;
    function Y = retraction_qr(X, U, t)
        % It is necessary to call qr_unique rather than simply qr to ensure
        % this is a retraction, to avoid spurious column sign flips.
        XU = multiprod(X, U);
        if nargin < 3
            Y = qr_unique(X + XU);
        else
            Y = qr_unique(X + t*XU);
        end
        % This is guaranteed to always yield orthogonal matrices with
        % determinant +1. Indeed: look at the eigenvalues of a skew
        % symmetric matrix, then at those of "identity plus that matrix",
        % and compute their product for the determinant: it is strictly
        % positive in all cases.
    end

    M.invretr_qr = @inverse_retraction_qr;
    function S = inverse_retraction_qr(X, Y)
        
        % Assume k = 1 in this explanation:
        % If Y = Retr_X(XS) where S is a skew-symmetric matrix, then
        %  X(I+S) = YR
        % for some upper triangular matrix R. Multiply with X' on the left:
        %  I + S = (X'Y) R    (*)
        % Since S is skew-symmetric, add the transpose of this equation:
        %  2I + 0 = (X'Y) R + R' (X'Y)'
        % We can solve for R by calling solve_for_triu, then solve for S
        % using equation (*).
        R = zeros(n, n, k);
        XtY = multiprod(multitransp(X), Y);
        H = 2*eye(n);
        for kk = 1 : k
            R(:, :, kk) = solve_for_triu(XtY(:, :, kk), H);
        end
        % In exact arithmetic, taking the skew-symmetric part has the
        % effect of subtracting the identity from each slice; in inexact
        % arithmetic, taking the skew-symmetric part is beneficial to
        % further enforce tangency.
        S = multiskew(multiprod(XtY, R));
        
    end
    
    % A second-order retraction is implemented here. To force its use,
    % after creating the factory M, execute M.retr = M.retr_polar.
    M.retr_polar = @retraction_polar;
    function Y = retraction_polar(X, U, t)
        if nargin == 3
            tU = t*U;
        else
            tU = U;
        end
        Y = X + multiprod(X, tU);
        for kk = 1 : k
            [Uk, ~, Vk] = svd(Y(:, :, kk));
            Y(:, :, kk) = Uk*Vk';
            % One can check that det(Uk*Vk') = det(X) for all skew-sym U.
            % That's because X + XU = X(I+U), and U is normal (UU' = U'U)
            % hence it can be unitarily diagonalized as U = WDW' with W
            % unitary and D diagonal; the eigenvalues of U are purely
            % imaginary because U+U' = 0, and they come in conjugate pairs
            % because U is real (there's a zero eigenvalue if n is odd).
            % This way, it follows that
            %   det(X+XU) = det(X)det(I+U) = det(X)det(I+D)
            % and det(I+D) is real, strictly positive provided n >= 2.
        end
    end

    M.invretr_polar = @inverse_retraction_polar;
    function S = inverse_retraction_polar(X, Y)
        
        % Assume k = 1 in this explanation:
        % If Y = Retr_X(XS) where S is a skew-symmetric matrix, then
        %  X(I+S) = YM
        % for some symmetric matrix M. Multiply with X' on the left:
        %  I + S = (X'Y) M    (*)
        % Since S is skew-symmetric, add the transpose of this equation:
        %  2I + 0 = (X'Y) M + M (X'Y)'
        % We can solve for M by calling sylvester, then solve for S
        % using equation (*).
        MM = zeros(n, n, k);
        XtY = multiprod(multitransp(X), Y);
        H = 2*eye(n);
        for kk = 1 : k
            MM(:, :, kk) = sylvester_nochecks(XtY(:, :, kk), XtY(:, :, kk)', H);
        end
        % In exact arithmetic, taking the skew-symmetric part has the
        % effect of subtracting the identity from each slice; in inexact
        % arithmetic, taking the skew-symmetric part is beneficial to
        % further enforce tangency.
        S = multiskew(multiprod(XtY, MM));
        
    end

    % By default, use QR retraction
    M.retr = M.retr_qr;
    M.invretr = M.invretr_qr;
    
    % For backward compatibility:
    M.retr2 = M.retr_polar;
    
    % Special case: for n = 3, we use Rodrigues formulas for expm / logm.
    if n == 3
        myexpm = @expm_SO3;
        mylogm = @logm_SO3;
    else
        myexpm = @expm;
        mylogm = @logm;
    end

    M.exp = @exponential;
    function Y = exponential(X, U, t)
        if nargin == 3
            exptU = t*U;
        else
            exptU = U;
        end
        for kk = 1 : k
            exptU(:, :, kk) = myexpm(exptU(:, :, kk));
        end
        Y = multiprod(X, exptU);
    end
    
    M.log = @logarithm;
    function U = logarithm(X, Y)
        U = multiprod(multitransp(X), Y);
        for kk = 1 : k
            % The result of logm should be real in theory, but it is
            % numerically useful to force it.
            U(:, :, kk) = real(mylogm(U(:, :, kk)));
        end
        % Ensure the tangent vector is in the Lie algebra.
        U = multiskew(U);
    end

    M.hash = @(X) ['z' hashmd5(X(:))];
    
    M.rand = @() randrot(n, k);
    
    M.randvec = @randomvec;
    function U = randomvec(X) %#ok<INUSD>
        U = randskew(n, k);
        nrmU = sqrt(U(:).'*U(:));
        U = U / nrmU;
    end
    
    M.lincomb = @matrixlincomb;
    
    M.zerovec = @(x) zeros(n, n, k);
    
    % Cheap vector transport
    M.transp = @(x1, x2, d) d;
    % This transporter is isometric (but it is /not/ parallel transport
    % along geodesics.)
    M.isotransp = M.transp;
    
    M.pairmean = @pairmean;
    function Y = pairmean(X1, X2)
        V = M.log(X1, X2);
        Y = M.exp(X1, .5*V);
    end
    
    M.dist = @(x, y) M.norm(x, M.log(x, y));
    
    M.vec = @(x, u_mat) u_mat(:);
    M.mat = @(x, u_vec) reshape(u_vec, [n, n, k]);
    M.vecmatareisometries = @() true;

end

% The following code is based on a proposal by Spencer Kraisler, June 2022.
% https://groups.google.com/g/manopttoolbox/c/KwdpyLiPUBw/m/aS-Yjq-pAwAJ
%
% Rodrigues formula for matrix exponential on SO(3): R = expm(phi)
function R = expm_SO3(phi)
    phi_vee = [-phi(2, 3); phi(1, 3); -phi(1, 2)];
    norm_phi_vee = norm(phi_vee);
    if norm_phi_vee > 0
        q1 = sin(norm_phi_vee)/norm_phi_vee;
        r = norm_phi_vee / 2;
        q2 = (sin(r)/r).^2 / 2;
        R = eye(3) + q1*phi + q2*phi^2;
    else
        R = eye(3);
    end
end
% Rodrigues formula for inverting matrix exp on SO(3): phi = logm(R)
function phi = logm_SO3(R)
    t = trace(R);
    norm_t = real(acos((t - 1)/2));
    if norm_t > 0 % could fail even when trace(R) < 3, because sensitive
        q = .5*norm_t/sin(norm_t);
    else
        q = .5;   % even with this, phi (below) could be nonzero
    end
    phi = q * [R(3, 2) - R(2, 3); R(1, 3) - R(3, 1); R(2, 1) - R(1, 2)];
    phi = [0 -phi(3) phi(2); phi(3) 0 -phi(1); -phi(2) phi(1) 0];
end
