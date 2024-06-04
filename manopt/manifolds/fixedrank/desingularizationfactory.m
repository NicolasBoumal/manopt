function M = desingularizationfactory(m, n, r, alpha)
% Manifold to optimize over bounded-rank matrices with an embedded geometry
%
% function M = desingularizationfactory(m, n, r)
%
% Implements the (smooth) desingularization geometry for the (nonsmooth)
% set of matrices of size mxn with rank <= r.
% This was first proposed in
%     Desingularization of bounded-rank matrix sets,
%     by Valentin Khrulkov and Ivan Oseledets
%     https://arxiv.org/abs/1612.03973
% then refined in an upcoming paper by the authors of this file.
% The geometry implemented here matches that second paper.
%
% The embedding space is E = R^(mxn) x Sym(n) where Sym denotes the set of
% symmetric matrices. Let Gr(n, s) be the Grassmannian: orthogonal
% projectors of size n and of rank s.
% The desingularization manifold is formally defined as
%
%     M = {(X, P) in E such that P is in Gr(n, n - r) and XP = 0}.
%
% The condition XP = 0 implies that X has rank at most r.
%
% A point (X, P) in M is represented as a structure with three fields:
%
%     U, S, V  such that  X = U*S*V'  and  P = I - V*V'.
%
% Matrices U (mxr) and V (nxr) are orthonormal, while S (rxr) is diagonal.
%
% A tangent vector at (X, P) is represented as a structure with two fields:
%
%     K, Vp  such that  Xdot = K*V' + U*S*Vp'  and  Pdot = -Vp*V' - V*Vp'.
% 
% The matrix K (mxr) is arbitrary while Vp (nxr) satisfies Vp' * V = 0.
%
% We equip the embedding space E with the metric
%
%     inner((Xd1, Pd1), (Xd2, Pd2)) = Tr(Xd1'*Xd2) + alpha * Tr(Pd1'*Pd2)
%
% for some parameter alpha (the default is 1/2).
%
%   TODO: explain how objects in the embedding space are represented.
%
% The tangent spaces of M inherit this metric, so that M is a Riemannian
% submanifold of E.
% 
% See also: fixedrankembeddedfactory

% This file is part of Manopt: www.manopt.org.
% Original authors: Quentin Rebjock and Nicolas Boumal, May 2024.
% Contributors:
% Change log:

    if ~exist('alpha', 'var') || isempty(alpha)
        alpha = .5;
    end
    
    assert(r <= min(m, n), 'The rank r should be <= min(m, n).');
    assert(alpha > 0, 'alpha should be positive (default is 1/2).');

    M.alpha = alpha;

    M.name = @() sprintf(['Desingularization manifold of '
                          '%dx%d matrices with rank bounded '
                          'by %d with alpha = %g'], m, n, r, alpha);
    
    M.dim = @() (m + n - r)*r;

    sfactor = @(X) 2*alpha*eye(r) + X.S^2;
    M.sfactor = sfactor;
    M.sfactorinv = @(X) diag(1./diag(sfactor(X)));
    
    % Usual trace inner product of two matrices.
    matinner = @(A, B) A(:)'*B(:);

    M.inner = @(X, Xd1, Xd2) matinner(Xd1.K, Xd2.K) + ...
                             matinner(Xd1.Vp, Xd2.Vp*sfactor(X));
    
    M.norm = @(X, Xd) sqrt(max(0, M.inner(X, Xd, Xd)));
    
    M.dist = @(X, Y) error('desingularization dist not implemented yet.');
    
    M.typicaldist = @() M.dim();
    
    % Given Xd in tangent vector format, projects the component Vp such
    % that it satisfies the tangent space constraints up to numerical
    % errors.
    % If Xd was indeed a tangent vector at X, this should barely affect Xd
    % (it would not at all if we had infinite numerical accuracy).
    M.tangent = @tangent;
    function Xd = tangent(X, Xd)
        Xd.Vp = Xd.Vp - X.V*(X.V'*Xd.Vp);
    end
    
    % Z is in the embedding space, that is, it is a struct with fields:
    %   Z.Y  --  an mxn matrix
    %   Z.Z  --  an nxn matrix
    % This function projects Z to the tangent space at X.
    % The output is in the tangent vector format.
    M.proj = @projection;
    function Zproj = projection(X, Z)
        % TBD: which is more efficient in practice?
        % symZV = Z.Z*X.V + Z.Z'*X.V;
        symZV = (Z.Z + Z.Z')*X.V;
        B = (Z.Y'*X.U*X.S - alpha*symZV) / sfactor(X);
        
        Zproj.K = Z.Y*X.V;
        Zproj.Vp = B - X.V*(X.V'*B);
    end

    % egrad is .....
    % rgrad is a tangent vector, in the tangent vector format.
    % TODO: clarify here and in the help section whether egrad is
    %       expected to be the gradient on E wrt its inner product (with
    %       alpha) -- which would be the default assumption -- or (as is
    %       more likely the case) that egrad is the gradient "downstairs"
    %       (which would require some explanations).
    M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(X, egrad)
        B = (egrad'*X.U*X.S) / sfactor(X);
        
        rgrad.K = egrad*X.V;
        rgrad.Vp = B - X.V*(X.V'*B);
    end

    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, H)
        S = sfactor(X);

        Q = eye(m) - X.U*(X.S^2/S)*X.U';
        B = (ehess'*X.U*X.S + egrad'*Q*H.K) / S;

        rhess.K = ehess*X.V + Q*egrad*H.Vp;
        rhess.Vp = B - X.V*(X.V'*B);
    end
    
    % Multiple retractions are available for the desingularization.
    % Default: retraction based on Q-factor.
    M.retr = @qfactor_retr;

    M.qfactor_retr = @qfactor_retr;
    function Y = qfactor_retr(X, Xd, t)
        if nargin < 3
            t = 1;
        end
        [Q, ~] = qr(X.V + t*Xd.Vp, 0);
        W = (X.U*X.S + t*Xd.K)*(X.V'*Q) + t*X.U*X.S*(Xd.Vp'*Q);
        
        [YU, YS, H] = svd(W, 'econ');
        Y.U = YU;
        Y.S = YS;
        Y.V = Q * H;
    end

    % Metric projection retraction: take a step in the ambient space and
    % project back to the desingularization.
    M.metric_proj = @metric_proj;
    function Y = metric_proj(X, Xd, t)
        if nargin < 3
            t = 1;
        end
        
        S2 = X.S.^2;
        KtUS = t*Xd.K'*X.U*X.S;
        D12 = S2 + KtUS + 2*alpha*eye(r);
        D = [
            S2 + KtUS + KtUS' + t^2*Xd.K'*Xd.K + 2*alpha*eye(r), D12;
            D12', S2
            ];

        [Qv, Rv] = qr([X.V, t*Xd.Vp], 0);
        W = Rv*D*Rv';
        W = (W' + W)/2;
        
        [Uw, Lams] = eig(W);
        [~, ind] = sort(diag(Lams), 'descend');
        Uw = Uw(:, ind);
        Vtilde = Qv*Uw(:, 1:r);
        
        AVtilde = (X.U*X.S + t*Xd.K) * (X.V'*Vtilde) + t*X.U*X.S*(Xd.Vp'*Vtilde);
        
        [YU, YS, H] = svd(AVtilde, 'econ');
        Y.U = YU;
        Y.S = YS;
        Y.V = Vtilde*H;
    end

    % Second-order retraction based on the polar retraction.
    M.polar = @polar;
    function Y = polar(X, Xd, t)
        if nargin < 3
            t = 1;
        end
        Z = X.V + t*Xd.Vp*(eye(r) - (t*Xd.K'*X.U*X.S) / sfactor(X));
        Vtilde = Z/sqrtm(Z'*Z);
        
        AVtilde = (X.U*X.S + t*Xd.K)*(X.V'*Vtilde) + t*X.U*X.S*(Xd.Vp'*Vtilde);
        
        [YU, YS, H] = svd(AVtilde, 'econ');
        Y.U = YU;
        Y.S = YS;
        Y.V = Vtilde * H;
    end

    % Same hash as fixedrankembeddedfactory.
    M.hash = fixedrankembeddedfactory(m, n, r).hash;
    
    % Generate a random point on M.
    % The factors U and V are sampled uniformly at random on Stiefel.
    % The singular values are uniform in [0, 1].
    M.rand = @random;
    function X = random()
        X.U = qr_unique(randn(m, r));
        X.V = qr_unique(randn(n, r));
        X.S = diag(sort(rand(r, 1), 'descend'));
    end
    
    % Generate a unit-norm random tangent vector at X.
    % Note: this may not be the uniform distribution.
    M.randvec = @randomvec;
    function Xd = randomvec(X)
        Xd.K  = randn(m, r);
        Xd.Vp = randn(n, r);
        Xd = tangent(X, Xd);
        normXd = M.norm(X, Xd);
        Xd.K = Xd.K/normXd;
        Xd.Vp = Xd.Vp/normXd;
    end
    
    % Linear combination of tangent vectors.
    % Returns the tangent vector a1*Xd1 + a2*Xd2.
    M.lincomb = @lincomb;
    function d = lincomb(~, a1, Xd1, a2, Xd2)
        if nargin == 3
            d.K = a1*Xd1.K;
            d.Vp = a1*Xd1.Vp;
        elseif nargin == 5
            d.K  = a1*Xd1.K + a2*Xd2.K;
            d.Vp = a1*Xd1.Vp + a2*Xd2.Vp;
        else
            error('desingularizationfactory.lincomb takes 3 or 5 inputs.');
        end
    end
    
    M.zerovec = @(X) struct('K', zeros(m, r), 'Vp', zeros(n, r));
    
    % The function 'vec' is isometric from the tangent space at X to real
    % vectors of length (m+n+r)r.
    M.vec = @vec;
    function Zvec = vec(X, Xd)
        VpS = Xd.Vp*sqrt(sfactor(X));
        Zvec = [Xd.K(:); VpS(:)];  % TODO: changed Xd to Xd.K : ok?
    end

    % The function 'mat' is the left-inverse of 'vec'. It is sometimes
    % useful to apply 'tangent' to the output of 'mat'.
    M.mat = @mat;
    function Xd = mat(X, v)
        K = reshape(v(1:(m*r)),  [m, r]);
        VpS = reshape(v((m*r)+(1:(n*r))), [n, r]);
        Xd = struct('K', K, 'Vp', VpS/sqrt(sfactor(X)));
    end
    
    M.vecmatareisometries = @() true;
    
    % It is sometimes useful to switch between representation of matrices
    % as triplets or as full matrices of size m x n. The function to
    % convert a matrix to a triplet, matrix2triplet, allows to specify the
    % rank of the representation. By default, it is equal to r. Omit the
    % second input (or set to inf) to get a full SVD triplet (in economy
    % format). If so, the resulting triplet does not represent a point on
    % the manifold.
    M.matrix2triplet = @matrix2triplet;
    function X_triplet = matrix2triplet(X_matrix, k)
        if ~exist('k', 'var') || isempty(k) || k <= 0
            k = r;
        end
        if k < min(m, n)
            [U, S, V] = svds(X_matrix, k);
        else
            [U, S, V] = svd(X_matrix, 'econ');
        end
        X_triplet.U = U;
        X_triplet.S = S;
        X_triplet.V = V;
    end

    M.triplet2matrix = @triplet2matrix;
    function X_matrix = triplet2matrix(X_triplet)
        U = X_triplet.U;
        S = X_triplet.S;
        V = X_triplet.V;
        X_matrix = U*S*V';
    end

    % Embed the point to the ambient space.
    % TODO: needs better documentation after details are fixed
    M.embed = @embed;
    function [Xe, Pe] = embed(X)
        Xe = X.U*X.S*X.V';
        Pe = eye(n) - X.V*X.V';
    end

    % Embed a tangent vector to the ambient space.
    % TODO: is the purpose of this different from tangent2ambient?
    M.embedtangent = @embedtangent;
    function [Xde, Pde] = embedtangent(X, Xd)
        Xde = Xd.K*X.V' + X.U*X.S*Xd.Vp';
        Pde = -Xd.Vp*X.V' - X.V*Xd.Vp';
    end

    % Transforms a tangent vector Z represented as a structure (K, Vp)
    % into a structure with fields (U, S, V) that represents that same
    % tangent vector in the ambient space of mxn matrices, as U*S*V'.
    % This matrix is equal to X.U*Z.M*X.V' + Z.Up*X.V' + X.U*Z.Vp'. The
    % latter is an mxn matrix, which could be too large to build
    % explicitly, and this is why we return a low-rank representation
    % instead. Note that there are no guarantees on U, S and V other than
    % that USV' is the desired matrix. In particular, U and V are not (in
    % general) orthonormal and S is not (in general) diagonal.
    % (In this implementation, S is identity, but this might change.)
    M.tangent2ambient_is_identity = false;
    M.tangent2ambient = @tangent2ambient;
    function Zambient = tangent2ambient(X, Z)
        % TODO: Call these fields Xdot and Pdot rather than Y and Z?
        %       Friction with M.proj?
        Zambient.Y = Z.K*X.V' + X.U*X.S*Z.Vp';
        Zambient.Z = -Z.Vp*X.V' - X.V*Z.Vp';
    end

    M.pmetricproj_retr = @pmetricproj_retr;
    function Y = pmetricproj_retr(X, Z, t)
        if nargin < 3
            t = 1;
        end
        [Qv, Rv] = qr([X.V, t * Z.Vp], 0);
        A = [eye(r), eye(r); eye(r), zeros(r)];
        H = Rv * A * Rv';
        H = (H' + H)/2;
        % [Ur, ~, ~] = svd(H);
        [Ur2, D] = eig(H);
        [~, ind] = sort(diag(D), 'descend');
        Ur2 = Ur2(:, ind);
        Ur = Ur2;
        
        prod = Qv * Ur;
        Vtilde = prod(:, 1:r);
        
        P1 = X.V' * Vtilde;
        P2 = t * Z.Vp' * Vtilde;
        H = (X.U * X.S + t * Z.K) * P1 + X.U * X.S * P2;
        
        [Uh, Sh, Vh] = svd(H, 'econ');
        Y.U = Uh;
        Y.S = Sh;
        Y.V = Vtilde * Vh;
    end

end
