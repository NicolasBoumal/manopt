function M = desingularizationfactory(m, n, r, alpha)
% Manifold to optimize over bounded-rank matrices through desingularization
%
% function M = desingularizationfactory(m, n, r)
%
% Implements the (smooth) desingularization geometry for the (nonsmooth)
% set of real matrices of size mxn with rank <= r.
% 
% This was first proposed in
%     Desingularization of bounded-rank matrix sets,
%     by Valentin Khrulkov and Ivan Oseledets
%     https://arxiv.org/abs/1612.03973
% then refined in an upcoming paper by the authors of this file.
% The geometry implemented here matches that second paper.
%
% The embedding space is E = R^(mxn) x Sym(n) where Sym denotes the set of
% symmetric matrices.
% Let Gr(n, s) be the Grassmannian: orthogonal projectors of size n and of
% rank s. The desingularization manifold is formally defined as
%
%     M = {(X, P) in E such that P is in Gr(n, n - r) and X*P = 0}.
%
% The condition X*P = 0 implies that X has rank at most r.
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
% The matrix K (mxr) is arbitrary while Vp (nxr) satisfies V'*Vp = 0.
%
% We equip the embedding space E with the metric
%
%     inner((Xd1, Pd1), (Xd2, Pd2)) = Tr(Xd1'*Xd2) + alpha * Tr(Pd1'*Pd2)
%
% for some parameter alpha > 0 (the default is 1/2).
%
% The tangent spaces of M inherit this metric, so that M is a Riemannian
% submanifold of E.
%
% The embedding space is potentially high dimensional, as it is composed of
% pairs of matrices: one of size mxn, the other of size nxn.
% To allow for efficient representation of matrices in the embedding space
% (e.g., for defining problem.egrad and problem.ehess), all objects in the
% embedding space are represented using the formats encoded in:
%
%   Emn = euclideanlargefactory(m, n);
%   Enn = euclideanlargefactory(n, n);
%
% For example, one can use sparse matrices of structures (U, S, V).
% See 'help euclideanlargefactory' for more.
%
% The desingularization M is a lift. Consider the problem of minimizing the
% function f over the variety of bounded-rank matrices. Define the map
% phi(X, P) = X over M and the composition g = f o phi. If a problem
% structure defines egrad and ehess, then those should provide the
% Euclidean gradient and Hessian of f seen as a function in all of R^(mxn)
% with the usual trace inner product. The tools egrad2rgrad and ehess2rhess
% built in this factory convert such information to Riemannian gradient and
% Hessian of g. Notice that this means g(X, P) = f(X) depends only on X,
% not on P. This is indeed the standard use case. If however you need to
% optimize a function of both X and P, then egrad2rgrad and ehess2rhess
% cannot be used. It is then necessary to provide the Riemannian
% derivatives in the problem structure (grad/hess, not egrad/ehess).
%
% Multiple retractions are available:
% - retr_qfactor (first-order).
% - retr_metric_proj (second-order).
% - retr_polar (second-order).
% The defaults are M.retr = @retr_qfactor and M.retr2 = @retr_polar.
%
% See also: fixedrankembeddedfactory euclideanlargefactory

% This file is part of Manopt: www.manopt.org.
% Original authors: Quentin Rebjock and Nicolas Boumal, June 2024.
% Contributors:
% Change log:
%   July 2, 2024 (NB)
%       Made M.retr2 = M.retr_polar available, to mark it as second order.

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

    % The embedding space consists of potentially large matrices.
    % We use euclideanlargefactory to allow efficient representations.
    Emn = euclideanlargefactory(m, n);
    Enn = euclideanlargefactory(n, n);


    M.alpha = alpha;

    M.name = @() sprintf(['Desingularization manifold of '
                          '%dx%d matrices with rank bounded '
                          'by %d with alpha = %g'], m, n, r, alpha);

    M.dim = @() (m + n - r)*r;

    sfactor = @(XP) 2*alpha*eye(r) + XP.S^2;
    sfactorinv = @(XP) diag(1./diag(sfactor(XP)));
    M.sfactor = sfactor;
    M.sfactorinv = sfactorinv;

    % Usual trace inner product of two matrices.
    matinner = @(A, B) A(:)'*B(:);

    M.inner = @(XP, XPdot1, XPdot2) matinner(XPdot1.K, XPdot2.K) + ...
                                matinner(XPdot1.Vp, XPdot2.Vp*sfactor(XP));

    M.norm = @(XP, XPdot) sqrt(max(0, M.inner(XP, XPdot, XPdot)));

    M.typicaldist = @() M.dim();

    % Given XPdot in tangent vector format, projects the component Vp such
    % that it satisfies the tangent space constraints up to numerical
    % errors.
    % If XPdot was indeed a tangent vector at XP, this should barely affect
    % XPdot (it would not at all if we had infinite numerical accuracy).
    M.tangent = @tangent;
    function XPdot = tangent(XP, XPdot)
        XPdot.Vp = XPdot.Vp - XP.V*(XP.V'*XPdot.Vp);
    end

    % XPa is in the embedding space E, that is, it is a struct with fields:
    %   XPa.X  --  an mxn matrix in euclideanlargefactory format, Emn
    %   XPa.P  --  an nxn matrix in euclideanlargefactory format, Enn
    % This function projects XPa to the tangent space at XP.
    % The output is in the tangent vector format.
    M.proj = @projection;
    function XPdot = projection(XP, XPa)
        % In principle, XPa.P should already be symmetric.
        % We take the symmetric part (times 2) to be permissive, but if
        % this becomes a performance issue there is something to gain here.
        % In matrix format, symPV equals (XPa.P + XPa.P.')*XP.V;
        symPV = Enn.times(XPa.P, XP.V) + Enn.transpose_times(XPa.P, XP.V);

        % In matrix format, the first term is XPa.X.'*XP.U*XP.S;
        B = Emn.transpose_times(XPa.X, XP.U*XP.S) - alpha*symPV;
        B = B / sfactor(XP);

        XPdot.K = Emn.times(XPa.X, XP.V); % = XPa.X*XP.V in matrix format
        XPdot.Vp = B - XP.V*(XP.V.'*B);
    end

    % Let f be the function R^{mxn} -> R to optimize over the
    % bounded-rank matrices.
    % Below, egrad is the gradient of f and
    % ehess is the Hessian of f in direction H.
    % Both egrad and ehess are in the ambient space format Emn.

    % rgrad is a tangent vector, in the tangent vector format.
    % It is the projection of (egrad, 0) onto the tangent space at (X, P).
    % 
    M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(XP, egrad)
        B = Emn.transpose_times(egrad, XP.U*XP.S) / sfactor(XP);
        rgrad.K = Emn.times(egrad, XP.V);
        rgrad.Vp = B - XP.V*(XP.V.'*B);
    end

    % rhess and H are tangent vectors, in the tangent vector format.
    % As above, egrad and ehess are in the ambient format.
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(XP, egrad, ehess, H)
        Sf = sfactor(XP);
        US = XP.U*(XP.S^2/Sf);

        B = (Emn.transpose_times(ehess, XP.U*XP.S) + ...
             Emn.transpose_times(egrad, (H.K - US*(XP.U.'*H.K)))) / Sf;

        GVp = Emn.times(egrad, H.Vp);
        rhess.K = Emn.times(ehess, XP.V) + GVp - US*(XP.U.'*GVp);
        rhess.Vp = B - XP.V*(XP.V.'*B);
    end

    % XP represents the current point (X, P). XPdot represents a tangent
    % vector. Let Pnew = P + Pdot. Vnew is such that Pnew = I - Vnew*Vnew'.
    % Compute a representation of ((X + Xdot)(I - Pnew), Pnew).
    % The optional parameter t multiplies the tangent vector XPdot.
    function XPnew = ambientstep2M(XP, XPdot, Vnew, t)
        if nargin < 4
            t = 1;
        end
        W = (XP.U*XP.S + t*XPdot.K)*(XP.V.'*Vnew) + ...
                                             t*XP.U*XP.S*(XPdot.Vp.'*Vnew);
        % W has size m-by-r
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
    % project back to the desingularization manifold.
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
        KtUS = (XPdot.K'*XP.U*XP.S) / sfactor(XP);
        Z = XP.V + t*XPdot.Vp*(eye(r) - t*KtUS);
        Vnew = Z/sqrtm(Z'*Z);

        XPnew = ambientstep2M(XP, XPdot, Vnew, t);
    end

    % Multiple retractions are available for the desingularization.
    % We choose default first- and second-order retractions here.
    M.retr = M.retr_qfactor;
    M.retr2 = M.retr_polar;

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
    M.matrix2triplet = fixedrankembeddedfactory(m, n, r).matrix2triplet;
    M.triplet2matrix = fixedrankembeddedfactory(m, n, r).triplet2matrix;

    % Map the representation (U, S, V) of a point XP in M to the
    % corresponding element of the ambient space Emn x Enn.
    % Returns a struct XPa with two fields:
    %     XPa.X represents U*S*V'    of size mxn in Emn format
    %     XPa.P represents I - V*V'  of size nxn in Enn format
    M.point2ambient_is_identity = false;
    M.point2ambient = @point2ambient;
    function XPa = point2ambient(XP)
        XPa.X = XP;
        XPa.P.times = @(A) A - XP.V*(XP.V.'*A);
        XPa.P.transpose_times = XPa.P.times;
    end

    % Transform a tangent vector XPdot represented as a structure (K, Vp)
    % into a structure that represents that same tangent vector in the
    % ambient space E.
    % Returns a struct XPa with two fields:
    %     XPa.X represents XPdot.K*XP.V' + XP.U*XP.S*XPdot.Vp'  of size mxn
    %     XPa.P represents -XPdot.Vp*XP.V' - XP.V*XPdot.Vp'     of size nxn
    % The representations follow euclideanlargefactory formats Emn and Enn.
    M.tangent2ambient_is_identity = false;
    M.tangent2ambient = @tangent2ambient;
    function XPa = tangent2ambient(XP, XPdot)
        XPa.X = struct('L', [XPdot.K, XP.U*XP.S], ...
                       'R', [XP.V, XPdot.Vp]);
        XPa.P = struct('L', -[XPdot.Vp, XP.V], ...
                       'R',  [XP.V, XPdot.Vp]);
    end

    % Simple proj transporter for a tangent vector XP1dot from XP1 to XP2.
    % If this becomes important, the code could be unpacked to improve
    % efficiency (marginally).
    M.transp = @transporter;
    function XP2dot = transporter(XP1, XP2, XP1dot)
        XP2dot = projection(XP2, tangent2ambient(XP1, XP1dot));
    end

end
