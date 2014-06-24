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


% TODO: In particular, specify how vectors in the embedding space are
% represented. Authorize more than one?

    M.name = @() sprintf('Manifold of %dx%d matrices of rank %d', m, n, k);
    
    M.dim = @() (m+n-k)*k;
    
    M.inner = @(x, d1, d2) d1.M(:).'*d2.M(:) + d1.Up(:).'*d2.Up(:) ...
                                             + d1.Vp(:).'*d2.Vp(:);
    
    M.norm = @(x, d) sqrt(M.inner(x, d, d));
    
    M.dist = @(x, y) error('fixedrank.dist not implemented yet.');
    
    M.typicaldist = @() 10*k;
    
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
    
    M.retr = @retraction;
    function Y = retraction(X, Z, t)
        if nargin < 3
            t = 1.0;
        end

        % See personal notes June 28, 2012 (NB)
        [Qu Ru] = qr(Z.Up, 0);
        [Qv Rv] = qr(Z.Vp, 0);
        
        % Calling svds or svd should yield the same result, but BV
        % advoacted svd is more robust, and it doesn't change the
        % asymptotic complexity to call svd then trim rather than call
        % svds. Also, apparently Matlab calls ARPACK in a suboptimal way
        % for svds in this scenario.
        % [Ut St Vt] = svds([X.S+t*Z.M , t*Rv' ; t*Ru , zeros(k)], k);
        [Ut St Vt] = svd([X.S+t*Z.M , t*Rv' ; t*Ru , zeros(k)]);
        
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

    M.hash = @(X) ['z' hashmd5([X.U(:) ; X.S(:) ; X.V(:)])];
    
    M.rand = @random;
    % Factors U and V live on Stiefel manifolds, hence we will reuse
    % their random generator.
    stiefelm = stiefelfactory(m, k);
    stiefeln = stiefelfactory(n, k);
    function X = random()
        X.U = stiefelm.rand();
        X.V = stiefeln.rand();
        X.S = randn(k);
    end
    
    M.randvec = @randomvec;
    function Z = randomvec(X)
        Z.Up = randn(m, k);
        Z.Vp = randn(n, k);
        Z.M  = randn(k, k);
        Z = projection(X, Z);
        nrm = M.norm(X, Z);
        Z.Up = Z.Up / nrm;
        Z.Vp = Z.Vp / nrm;
        Z.M  = Z.M  / nrm;
    end
    
    M.lincomb = @lincomb;
    
    M.zerovec = @(X) struct('Up', zeros(m, k), 'M', zeros(k, k), ...
                                                        'Vp', zeros(n, k));
    
    M.transp = @(x1, x2, d) projection(x2, d);

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
        error('Bad use of fixedrank.lincomb.');
    end

end
