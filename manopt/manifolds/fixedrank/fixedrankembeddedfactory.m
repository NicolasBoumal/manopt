function M = fixedrankembeddedfactory(m, n, k)
% Manifold struct to optimize fixed-rank matrices w/ an embedded geometry.
%
% function M = fixedrankembeddedfactory(m, n, k)
%
% Warning: this code was little tested and should be used with care.
%
% Manifold of m-by-n matrices of fixed rank k. This follows the geometry
% described in the following paper (which for now is the documentation):
% B. Vandereycken, "Low-rank matrix completion by Riemannian optimization", 2011.
%
% Paper link: http://arxiv.org/pdf/1209.3834.pdf
%
% A point X on the manifold is represented as a structure with three
% fields: U, S and V. The matrices U (mxk) and V (nxk) are orthonormal,
% while the matrix S (kxk) is any full rank matrix.
%
% Tangent vectors are represented as a structure with three fields: Up, M
% and Vp. The matrices Up (mxk) and Vp (mxk) obey Up'*U = 0 and Vp'*V = 0.
% The matrix M (kxk) is arbitrary.
%
% The chosen geometry yields a Riemannian submanifold of the embedding
% space R^(mxn) equipped with the usual trace inner product.

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
%
%	Feb. 20, 2014 (NB):
%       Added function tangent to work with checkgradient.
%   June 24, 2014 (NB):
%       A couple modifications following
%       Bart Vandereycken's feedback:
%       - The checksum (hash) was replaced for a faster alternative: it's a
%         bit less "safe" in that collisions could arise with higher
%         probability, but they're still very unlikely.
%       - The vector transport was changed.
%       The typical distance was also modified, hopefully giving the
%       trustregions method a better initial guess for the trust region
%       radius, but that should be tested for different cost functions too.
%    July 11, 2014 (NB):
%       Added ehess2rhess and tangent2ambient, supplied by Bart.


% TODO: For the documentation, specify how vectors in the embedding space are
% represented, since more than one representation are allowed.

    M.name = @() sprintf('Manifold of %dx%d matrices of rank %d', m, n, k);
    
    M.dim = @() (m+n-k)*k;
    
    M.inner = @(x, d1, d2) d1.M(:).'*d2.M(:) + d1.Up(:).'*d2.Up(:) ...
                                             + d1.Vp(:).'*d2.Vp(:);
    
    M.norm = @(x, d) sqrt(M.inner(x, d, d));
    
    M.dist = @(x, y) error('fixedrankembeddedfactory.dist not implemented yet.');
    
    M.typicaldist = @() M.dim();
    
    M.tangent = @(X, Z) Z;
    
    M.proj = @projection;
    function Zproj = projection(X, Z)
        
        if isstruct(Z)
        
            % Projection for Z a vector in R^(mxn) represented as a
            % structure with the same format as that of points X on the
            % manifold. This is useful if Z is a low-rank matrix. The rank
            % of Z need not coincide with that of X.
            if isfield(Z, 'S')
                UtUz = X.U' * Z.U;
                VtVz = X.V' * Z.V;

                SzVztV = Z.S * VtVz';
                UtZV = UtUz * SzVztV;

                Zproj.M = UtZV;
                Zproj.Up = Z.U * SzVztV - X.U*UtZV;
                Zproj.Vp = Z.V * (UtUz*Z.S)' - X.V*UtZV';
                
            % Projection for Z a vector in R^(mxn) represented as a
            % structure with the same format as that of tangent vectors.
            elseif isfield(Z, 'M')
                
                Zproj.M = Z.M;
                Zproj.Up = Z.Up - X.U*(X.U'*Z.Up);
                Zproj.Vp = Z.Vp - X.V*(X.V'*Z.Vp);
                
            else
                error('fixedrank.proj: Bad ambient vector representation.');
            end
            
        % Projection for Z a vector in R^(mxn) represented as an mxn matrix
        else
            
            ZV = Z*X.V;
            UtZV = X.U'*ZV;
            ZtU = Z'*X.U;
            
            Zproj.M = UtZV;
            Zproj.Up = ZV  - X.U*UtZV;
            Zproj.Vp = ZtU - X.V*UtZV';
            
        end

    end

    M.egrad2rgrad = @projection;
    
    % Code supplied by Bart.
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, H)
        % egrad is a possibly sparse matrix of the same matrix size as X
        % ehess is the matvec of Euclidean Hessian with H, is also a matrix
        % of the same matrix size as X

        rhess = projection(X, ehess);

        % Curvature part            
        T = (egrad*H.Vp)/X.S;
        rhess.Up = rhess.Up + (T - X.U*(X.U'*T));

        T = (egrad'*H.Up)/X.S;
        rhess.Vp = rhess.Vp + (T - X.V*(X.V'*T));
    end

    % Transforms a tangent vector Z represented as a structure (Up, M, Vp)
    % into a matrix Xdot that corresponds to the same tangent vector, but
    % represented in the ambient space of matrices of size mxn.
    % Be careful though: the output is a full matrix of size mxn, which
    % could be prohibitively large in some applications. If that is so,
    % consider using the formulation with two outputs. The product of the
    % two outputs equals Xdot, but it is represented with smaller matrices.
    M.tangent2ambient = @tangent2ambient;
    function [O1, O2] = tangent2ambient(X, Z)
        if nargout == 1
            O1 = X.U*Z.M*X.V' + Z.Up*X.V' + X.U*Z.Vp';
        elseif nargout == 2
            O1 = [X.U*Z.M + Z.Up, X.U];
            O2 = [X.V' ; Z.Vp'];
        else
            error('fixedrankembeddedfactory.tangent2ambient can have either 1 or 2 outputs.');
        end
    end
    
    M.retr = @retraction;
    function Y = retraction(X, Z, t)
        if nargin < 3
            t = 1.0;
        end

        % See personal notes June 28, 2012 (NB)
        [Qu, Ru] = qr(Z.Up, 0);
        [Qv, Rv] = qr(Z.Vp, 0);
        
        % Calling svds or svd should yield the same result, but BV
        % advocated svd is more robust, and it doesn't change the
        % asymptotic complexity to call svd then trim rather than call
        % svds. Also, apparently Matlab calls ARPACK in a suboptimal way
        % for svds in this scenario.
        % [Ut St Vt] = svds([X.S+t*Z.M , t*Rv' ; t*Ru , zeros(k)], k);
        [Ut, St, Vt] = svd([X.S+t*Z.M , t*Rv' ; t*Ru , zeros(k)]);
        
        Y.U = [X.U Qu]*Ut(:, 1:k);
        Y.V = [X.V Qv]*Vt(:, 1:k);
        Y.S = St(1:k, 1:k);
        
        % equivalent but very slow code
        % [U S V] = svds(X.U*X.S*X.V' + t*(X.U*Z.M*X.V' + Z.Up*X.V' + X.U*Z.Vp'), k);
        % Y.U = U; Y.V = V; Y.S = S;
        
    end
    
    M.exp = @exponential;
    function Y = exponential(X, Z, t)
        if nargin < 3
            t = 1.0;
        end
        Y = retraction(X, Z, t);
        warning('manopt:fixedrank:exp', ['Exponential for fixed rank ' ...
                'manifold not implemented yet. Used retraction instead.']);
    end

    % Less safe but much faster checksum, June 24, 2014.
    % Older version right below.
    M.hash = @(X) ['z' hashmd5([sum(X.U(:)) ; sum(X.S(:)); sum(X.V(:)) ])];
    %M.hash = @(X) ['z' hashmd5([X.U(:) ; X.S(:) ; X.V(:)])];
    
    M.rand = @random;
    % Factors U and V live on Stiefel manifolds, hence we will reuse
    % their random generator.
    stiefelm = stiefelfactory(m, k);
    stiefeln = stiefelfactory(n, k);
    function X = random()
        X.U = stiefelm.rand();
        X.V = stiefeln.rand();
        X.S = diag(randn(k, 1));
    end
    
    M.randvec = @randomvec;
    function Z = randomvec(X)
        Z.Up = randn(m, k);
        Z.Vp = randn(n, k);
        Z.M  = diag(randn(k, 1));
        Z = projection(X, Z);
        nrm = M.norm(X, Z);
        Z.Up = Z.Up / nrm;
        Z.Vp = Z.Vp / nrm;
        Z.M  = Z.M  / nrm;
    end
    
    M.lincomb = @lincomb;
    
    M.zerovec = @(X) struct('Up', zeros(m, k), 'M', zeros(k, k), ...
                                                        'Vp', zeros(n, k));
    
    % New vector transport on June 24, 2014 (by Bart)
    M.transp = @project_tangent;
    function Zproj = project_tangent(x1, x2, d)
        Z.U = [x1.U*d.M+d.Up, x1.U]; 
        Z.S = eye(size(Z.U, 2));
        Z.V = [x1.V, d.Vp];
        Zproj = projection(x2, Z);
    end

end

% Linear combination of tangent vectors
function d = lincomb(x, a1, d1, a2, d2) %#ok<INUSL>

    if nargin == 3
        d.Up = a1*d1.Up;
        d.Vp = a1*d1.Vp;
        d.M  = a1*d1.M;
    elseif nargin == 5
        d.Up = a1*d1.Up + a2*d2.Up;
        d.Vp = a1*d1.Vp + a2*d2.Vp;
        d.M  = a1*d1.M  + a2*d2.M;
    else
        error('fixedrank.lincomb takes either 3 or 5 inputs.');
    end

end
