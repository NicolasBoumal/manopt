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
% The tangent spaces of M inherit this metric, so that M is a Riemannian
% submanifold of E.
%
% The desingularization M is a lift. Consider the problem of minimizing the
% function f over the variety of bounded-rank matrices. Define the map
% phi(X, P) = X over M and the composition g = f o phi.
%   TODO: explain how objects in the embedding space are represented.
%
% Multiple retractions are available for the desingularization.
% - retr_qfactor (first-order).
% - retr_metric_proj (second-order).
% - retr_polar (second-order).
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
    assert(alpha >= 0, 'alpha should be positive (default is 1/2).');
    if alpha == 0
        warning('desingularization:alphazero', ...
                ['alpha was set to 0. It should be positive.\n' ...
                'Disable this warning with ' ...
                'warning(''off'', ''desingularization:alphazero'').']);
    end

    M.alpha = alpha;

    M.name = @() sprintf(['Desingularization manifold of '
                          '%dx%d matrices with rank bounded '
                          'by %d with alpha = %g'], m, n, r, alpha);

    M.dim = @() (m + n - r)*r;

    sfactor = @(XP) 2*alpha*eye(r) + XP.S^2;
    M.sfactor = sfactor;
    M.sfactorinv = @(XP) diag(1./diag(sfactor(XP)));

    % Usual trace inner product of two matrices.
    matinner = @(A, B) A(:)'*B(:);

    M.inner = @(XP, XPdot1, XPdot2) matinner(XPdot1.K, XPdot2.K) + ...
                             matinner(XPdot1.Vp, XPdot2.Vp*sfactor(XP));

    M.norm = @(XP, XPdot) sqrt(max(0, M.inner(XP, XPdot, XPdot)));

    M.dist = @(XP1, XP2) error('desingularization dist not implemented yet.');

    M.typicaldist = @() M.dim();

    % Given XPdot in tangent vector format, projects the component Vp such
    % that it satisfies the tangent space constraints up to numerical
    % errors.
    % If XPdot was indeed a tangent vector at XP, this should barely affect XPdot
    % (it would not at all if we had infinite numerical accuracy).
    M.tangent = @tangent;
    function XPdot = tangent(XP, XPdot)
        XPdot.Vp = XPdot.Vp - XP.V*(XP.V'*XPdot.Vp);
    end

    % XPa is in the embedding space E, that is, it is a struct with fields:
    %   XPa.X  --  an mxn matrix
    %   XPa.P  --  an nxn matrix
    % This function projects XPa to the tangent space at XP.
    % The output is in the tangent vector format.
    M.proj = @projection;
    function XPdot = projection(XP, XPa)
        % Note the following about computing symPV:
        %  1) In principle, XPa.P should already be symmetric.
        %     We take (twice) the symmetric part to be safe.
        %  2) If XPa.P is full or sparse, the code below should work fine.
        %     If XPa.P is large and dense but it is possible to compute
        %     products of XPa.P with narrow matrices such as XP.V efficiently,
        %     then this code should be modified to take advantage of this.
        % symPV = XPa.P*XP.V + XPa.P'*XP.V;
        symPV = (XPa.P + XPa.P')*XP.V;
        B = (XPa.X'*XP.U*XP.S - alpha*symPV) / sfactor(XP);

        XPdot.K = XPa.X*XP.V;
        XPdot.Vp = B - XP.V*(XP.V'*B);
    end

    % Let f be the function R^{mxn} -> R to optimize over the
    % bounded-rank matrices. Below, egrad is the gradient of f and ehess
    % is the Hessian of f in direction H.

    % rgrad is a tangent vector, in the tangent vector format.
    % It is the projection of (egrad, 0) onto the tangent space at (X, P).
    M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(XP, egrad)
        B = (egrad'*XP.U*XP.S) / sfactor(XP);

        rgrad.K = egrad*XP.V;
        rgrad.Vp = B - XP.V*(XP.V'*B);
    end

    % rhess is a tangent vector, in the tangent vector format.
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(XP, egrad, ehess, H)
        S = sfactor(XP);
        US = XP.U*(XP.S^2/S);

        B = (ehess'*XP.U*XP.S + egrad'*(H.K - US*(XP.U'*H.K))) / S;

        GVp = egrad*H.Vp;
        rhess.K = ehess*XP.V + GVp - US*(XP.U'*GVp);
        rhess.Vp = B - XP.V*(XP.V'*B);
    end

    % Multiple retractions are available for the desingularization.
    % Default: retraction based on Q-factor.
    M.retr = @retr_qfactor;

    % XP represents the current point (X, P). XPdot represents a tangent
    % vector. Let Pnew = P + Pdot. Vnew is such that Pnew = I - Vnew*Vnew'.
    % Compute a representation of ((X + Xdot)(I - Pnew), Pnew).
    % The optional parameter t multiplies the tangent vector XPdot.
    function XPnew = ambientstep2M(XP, XPdot, Vnew, t)
      if nargin < 4
        t = 1;
      end
      W = (XP.U*XP.S + t*XPdot.K)*(XP.V'*Vnew) + t*XP.U*XP.S*(XPdot.Vp'*Vnew);

      [WU, WS, WV] = svd(W, 'econ');
      XPnew.U = WU;
      XPnew.S = WS;
      XPnew.V = Vnew * WV;
    end

    % First-order retraction based on Q-factor for Grassmann.
    M.retr_qfactor = @retr_qfactor;
    function XPnew = retr_qfactor(XP, XPdot, t)
      if nargin < 3
        t = 1;
      end
      [Vnew, ~] = qr(XP.V + t*XPdot.Vp, 0);

      XPnew = ambientstep2M(XP, XPdot, Vnew, t);
    end

    % Metric projection retraction: take a step in the ambient space and
    % project back to the desingularization.
    % This is a second-order retraction.
    M.retr_metric_proj = @retr_metric_proj;
    function XPnew = retr_metric_proj(XP, XPdot, t)
        if nargin < 3
            t = 1;
        end

        S2 = XP.S.^2;
        KtUS = t*XPdot.K'*XP.U*XP.S;
        D12 = S2 + KtUS + 2*alpha*eye(r);
        D = [
            S2 + KtUS + KtUS' + t^2*XPdot.K'*XPdot.K + 2*alpha*eye(r), D12;
            D12', S2
            ];

        [Qv, Rv] = qr([XP.V, t*XPdot.Vp], 0);
        W = Rv*D*Rv';
        W = (W' + W)/2;

        [Uw, Lams] = eig(W);
        [~, ind] = sort(diag(Lams), 'descend');
        Uw = Uw(:, ind);
        Vnew = Qv*Uw(:, 1:r);

        XPnew = ambientstep2M(XP, XPdot, Vnew, t);
    end

    % Second-order retraction based on the polar retraction.
    M.retr_polar = @retr_polar;
    function XPnew = retr_polar(XP, XPdot, t)
        if nargin < 3
            t = 1;
        end
        Z = XP.V + t*XPdot.Vp*(eye(r) - (t*XPdot.K'*XP.U*XP.S) / sfactor(XP));
        Vnew = Z/sqrtm(Z'*Z);

        XPnew = ambientstep2M(XP, XPdot, Vnew, t);
    end

    % Same hash as fixedrankembeddedfactory.
    M.hash = fixedrankembeddedfactory(m, n, r).hash;

    % Generate a random point on M.
    % The factors U and V are sampled uniformly at random on Stiefel.
    % The singular values are uniform in [0, 1].
    M.rand = @random;
    function XP = random()
        XP.U = qr_unique(randn(m, r));
        XP.V = qr_unique(randn(n, r));
        XP.S = diag(sort(rand(r, 1), 'descend'));
    end

    % Generate a unit-norm random tangent vector at XP.
    % Note: this may not be the uniform distribution.
    M.randvec = @randomvec;
    function XPdot = randomvec(XP)
        XPdot.K  = randn(m, r);
        XPdot.Vp = randn(n, r);
        XPdot = tangent(XP, XPdot);
        normXPdot = M.norm(XP, XPdot);
        XPdot.K = XPdot.K/normXPdot;
        XPdot.Vp = XPdot.Vp/normXPdot;
    end

    % Linear combination of tangent vectors.
    % Returns the tangent vector a1*XPdot1 + a2*XPdot2.
    M.lincomb = @lincomb;
    function XPdot3 = lincomb(~, a1, XPdot1, a2, XPdot2)
        if nargin == 3
            XPdot3.K = a1*XPdot1.K;
            XPdot3.Vp = a1*XPdot1.Vp;
        elseif nargin == 5
            XPdot3.K  = a1*XPdot1.K + a2*XPdot2.K;
            XPdot3.Vp = a1*XPdot1.Vp + a2*XPdot2.Vp;
        else
            error('desingularizationfactory.lincomb takes 3 or 5 inputs.');
        end
    end

    M.zerovec = @(XP) struct('K', zeros(m, r), 'Vp', zeros(n, r));

    % The function 'vec' is isometric from the tangent space at XP to real
    % vectors of length (m+n+r)r.
    M.vec = @vec;
    function XPdotvec = vec(XP, XPdot)
        VpS = XPdot.Vp*sqrt(sfactor(XP));
        XPdotvec = [XPdot.K(:); VpS(:)];
    end

    % The function 'mat' is the left-inverse of 'vec'. It is sometimes
    % useful to apply 'tangent' to the output of 'mat'.
    M.mat = @mat;
    function XPdot = mat(XP, v)
        K = reshape(v(1:(m*r)),  [m, r]);
        VpS = reshape(v((m*r)+(1:(n*r))), [n, r]);
        XPdot = struct('K', K, 'Vp', VpS/sqrt(sfactor(XP)));
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

    % Map the representation (U, S, V) of a point XP in M to the
    % corresponding element of the ambient space E.
    % Returns a struct XPa with two fields:
    %     XPa.X = U*S*V'    of size mxn
    %     XPa.P = I - V*V'  of size nxn
    M.embed = @point2ambient;
    function XPa = point2ambient(XP)
        XPa.X = XP.U*XP.S*XP.V';
        XPa.P = eye(n) - XP.V*XP.V';
    end

    % Transform a tangent vector XPdot represented as a structure (K, Vp)
    % into a structure that represents that same tangent vector in the
    % ambient space E.
    % Returns a struct XPa with two fields:
    %     XPa.X = XPdot.K*XP.V' + XP.U*XP.S*XPdot.Vp'  of size mxn
    %     XPa.P = -XPdot.Vp*XP.V' - XP.V*XPdot.Vp'     of size nxn
    M.tangent2ambient_is_identity = false;
    M.tangent2ambient = @tangent2ambient;
    function XPa = tangent2ambient(XP, XPdot)
        XPa.X = XPdot.K*XP.V' + XP.U*XP.S*XPdot.Vp';
        XPa.P = -XPdot.Vp*XP.V' - XP.V*XPdot.Vp';
    end

end
