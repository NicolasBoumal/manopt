function M = fixedranktensorembeddedfactory(tensor_size, tensor_rank)
% Manifold of tensors with fixed multilinear rank in Tucker format as a submanifold
% of a Euclidean space.
%
% function M = fixedranktensorembeddedfactory(tensor_size, tensor_rank)
%
% NOTE: Tensor Toolbox version 2.6 or higher is required for this factory:
% see https://www.tensortoolbox.org/ or https://gitlab.com/tensors/tensor_toolbox
%
% The set of tensors with fixed multilinear rank in Tucker format is endowed
% with a Riemannian manifold structure as an embedded submanifold of a
% Euclidean space. This function returns a structure M representing this
% manifold for use with Manopt.
%
% tensore_size is a vector ......... Please explain the input: what are the sizes / ranks?
% tensor_rank is a vector ..........
%
% A point on the manifold is represented as a structure with two fields:
% X: a ttensor object (see Tensor Toolbox), the actual point on the manifold.
% Cpinv: a cell list of the pseudoinverses of all matricizations of X.core.
% (This preprocessing makes subsequent operations more efficient.)
%
% A tangent vector is represented as a structure with two fields: 
% G: variation in the core tensor
% V: a cell list of variations in the core matrices
%
% For details, refer to the article
% "A Riemannian trust-region method for low-rank tensor completion"
% Gennadij Heidel and Volker Schulz, doi:10.1002/nla.2175.
%
% Please cite the Manopt and MATLAB Tensor Toolbox papers as well as the
% research paper:
%     @Article{heidel2018riemannian,
%       Title   = {A {R}iemannian trust-region method for low-rank tensor completion},
%       Author  = {G. Heidel and V. Schulz},
%       Journal = {Numerical Linear Algebra with Applications},
%       Year    = {2018},
%       Volume  = {23},
%       Number  = {6},
%       Pages   = {e1275},
%       Doi     = {10.1002/nla.2175}
%     }
%
% See also: fixedrankfactory_tucker_preconditioned fixedrankembeddedfactory

% This file is part of Manopt: www.manopt.org.
% Original author: Gennadij Heidel, January 24, 2019.
% Contributors: 
% Change log:

    assert(exist('ttensor','file')==2,['It seems the Tensor Toolbox is ',...
    'not installed. It is needed for the execution of fixedranktensorembeddedfactory.m. ',...
    'Please download the Toolbox first at https://www.tensortoolbox.org/',...
    ' or https://gitlab.com/tensors/tensor_toolbox.']);

    % Tensor size and rank
    d = length(tensor_size);
    if d~=length(tensor_rank)
        error('Bad usage of embedded_tensor_factory. Tensor dimensions and rank do not match.')
    end
    n = tensor_size;
    r = tensor_rank;
    
    % Generate a string that describes the used manifold
    M.name = @mfname;
    function spf = mfname()
        s = 'C';
        for i=1:d
            s = strcat(s,' x U',int2str(i));
        end
        s = strcat(s,' Tucker manifold of ');
        for i=1:10
           if n(i)<10^i
               digits = i;
               break;
           end
        end
        s = strcat(s,'%',int2str(digits+1),'d-by-');
        for i=2:d-1
            s = strcat(s,'%d-by-');
        end
        s = strcat(s,'%d tensors of rank ');
        for i=1:10
           if r(i)<10^i
               digits = i;
               break;
           end
        end
        s = strcat(s,'%',int2str(digits+1),'d-by-');
        for i=2:d-1
            s = strcat(s,'%d-by-');
        end
        s = strcat(s,'%d.');
        spf = sprintf(s, n, r);
    end
    
    M.dim = @mfdim;
    function mfd = mfdim()
        mfd = prod(r);
        for i=1:d
            mfd = mfd + n(i)*r(i) - r(i)^2;
        end
    end
    
    % Efficient inner product on tangent space exploiting orthogonality
    % relations
    M.inner = @iproduct;
    function ip = iproduct(X, eta, zeta)
        ip = innerprod(eta.G,zeta.G);
        for i=1:d
            ip = ip + innerprod(X.X.core,ttm(X.X.core,eta.V{i}'*zeta.V{i},i));
        end
    end

    M.norm = @(X, eta) sqrt(M.inner(X, eta, eta));
    
    M.dist = @(x, y) error('embedded_tensor_factory.dist not implemented yet.');
    
    M.typicaldist = @() 10*mean(n)*mean(r); % To do  
    
    % Riemannian gradient is the projection of the Euclidean gradient
    M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(X, egrad)
        rgrad = M.proj(X,egrad);
    end
    
    % Riemannian Hessian is the projection of the Euclidean Hessian plus a
    % curvature term
    M.ehess2rhess = @ehess2rhess;
    function Hess = ehess2rhess(X, egrad, ehess, eta) 
        Hess = lincomb(X, 1,M.proj(X,ehess), 1, curvature_term(egrad, X, eta));
    end
    
    M.proj = @projection;
    function Eproj = projection(X, E)
        if ~isstruct(E)
            uList = X.X.U;
            
            % cf. Kressner et al., p. 454, bottom
            G = ttm(E,uList,'t');

            % cf. Kressner et al., p. 454, bottom
            V = cell(1,d);
            for i = 1:d
                % modes vector witout ith mode for ttm multiplication
                modes = 1:d;
                modes(i) = [];

                % list of basis matrices U without the index i
                uListWoI = uList;
                uListWoI(:,i) = [];
                
                % the term of V_i before the multiplication by the orthogonal projector
                beforeProj = tenmat( ttm(E,uListWoI,modes,'t'), i) * X.Cpinv{i};

                % orthogonal projection
                V{i} = double(beforeProj - X.X.U{i}*(X.X.U{i}'*beforeProj));
            end
            
            Eproj.G = G;
            Eproj.V = V;
        else
            error('embedded_tensor_factory.proj only implemented for ambient tensors so far.');
        end
        
    end

    % Re-orthogonalise basis matrix variations in the tangent vector
    % When applied to a tangent vector, this should to nothing up to
    % numerical noise
    M.tangent = @tangent;
    function xi = tangent(X, eta)
        xi = eta;
        for i = 1:d
            xi.V{i} = eta.V{i} - X.X.U{i}*(X.X.U{i}'*eta.V{i});
        end
    end

    % Generate full n1-by-...-by-nd tensor in the ambient space from
    % tangent vector
    M.tangent2ambient = @tan2amb;
    function E = tan2amb(X,eta)
        E = ttm(eta.G,X.X.U);
        
        for i=1:d
            % modes vector witout ith mode for ttm multiplication
            modes = 1:d;
            modes(i) = [];

            % list of basis matrices U without the index i
            uListWoI = X.X.U;
            uListWoI(:,i) = [];

            E = E + ttm( ttm(X.X.core,eta.V{i},i), uListWoI, modes );
        end
        
    end
    
    % Efficient retraction, see Kressner at al., 2014, for idea
    M.retr = @retraction;
    function Y = retraction(X, xi, alpha)
        % if no alpha is given, assume it to be = 1
        if nargin < 3
            alpha = 1.0;
        end

        Q = cell(1,d);
        R = cell(1,d);
        for i=1:d
            [Q{i}, R{i}] = qr( [X.X.U{i}, xi.V{i}], 0 );
        end

        S = zeros(2*r);

        % First block C+alpha*G, see Kressner et al., Fig. 2
        sBlock = double(X.X.core + alpha*xi.G);
        for i=1:d
            sBlock = cat(i, sBlock, zeros(size(sBlock)));
        end
        S = S + sBlock;

        % Adjacent Blocks alpha*C, see Kressner et al., Fig. 2
        for i=1:d
            sBlock = double(alpha*X.X.core);
            modes = 1:d;
            modes(i) = [];
            for j=modes
                sBlock = cat(j, sBlock, zeros(size(sBlock)));
            end
            sBlock = cat(i, zeros(size(sBlock)), sBlock);
            S = S + sBlock;
        end

        % no concatenation in tensor toolbox --> detour over Matlab arrays
        S = tensor(S);

        % absorb R factors in core tensor
        S = ttm(S, R);

        % actual retraction
        sHosvd = hosvd(S, r);

        % absorb Q factors in basis matrices
        U = cell(1,d);
        for i=1:d
            U{i} = Q{i} * sHosvd.U{i};
        end

        Z = ttensor( sHosvd.core, U);
        Y.X = Z;
        Y.Cpinv = cell(1,d);
        for i = 1:d
           Y.Cpinv{i} =  pinv(double(tenmat(Y.X.core,i)));
        end
    end
    
    M.hash = @hashing;
    function h = hashing(X)
        v = zeros(2*d+1,1);
        for i=1:d
            v(i) = sum(X.X.U{i}(:));
        end
        v(d+1) = sum(X.X.core(:));
        for i=1:d
            v(d+1+i) = sum(X.Cpinv{i}(:));
        end
        h = ['z' hashmd5(v)];
    end
    
    % Random tensor on manifold
    M.rand = @random;
    function X = random()
        U = cell(1,d);
        R = cell(1,d);
        Q = cell(1,d);
        for i=1:d
            [U{i}, R{i}] = qr(rand(n(i),r(i)), 0);
        end
        C  = tenrand(r);
        C = ttm(C,R);
        
        % Peform an HSOVD of the core to ensure all-orthogonality
        Z = hosvd(C,r);
        for i=1:d
            U{i} = U{i}*Z.U{i};
        end
        
        Y = ttensor(Z.core,U);
        X.X = Y;
        Cpinv = cell(1,d);
        for i=1:d
            Cpinv{i} = pinv(double(tenmat(X.X.core,i)));
        end
        X.Cpinv = Cpinv;
    end
    
    % Random unit norm tangent vector
    M.randvec = @randomvec;
    function eta = randomvec(X)
        G = tensor(randn(r));
        xi.G = G;
        
        V = cell(1,d);
        for i=1:d
            V{i} = randn(n(i),r(i));
        end
        xi.V = V;
        
        xi = M.tangent(X,xi);
        nrm = M.norm(X,xi);
        
        eta.G = xi.G / nrm;
        for i=1:d
            xi.V{i} = xi.V{i} / nrm;
        end
        eta.V = xi.V;
    end
    
    % Evalueate lambda1*eta1* + lambda2*eta2 in the tangent space
    M.lincomb = @lincomb;
    function xi = lincomb(X, lambda1, eta1, lambda2, eta2)
        if nargin == 3
            V = cell(1,d);
            for i=1:d
                V{i} = lambda1*eta1.V{i};
            end
            xi.G = lambda1*eta1.G;
            xi.V = V;
        elseif nargin == 5
            V = cell(1,d);
            for i=1:d
                V{i} = lambda1*eta1.V{i} + lambda2*eta2.V{i};
            end
            xi.G = lambda1*eta1.G + lambda2*eta2.G;
            xi.V = V;
        else
            error('Bad use of embedded_tensor_factory.lincomb.');
        end
    end
    
    M.zerovec = @zerovector;
    function eta = zerovector(X)
        G = tenzeros(r);
        V = cell(1,d);
        for i=1:d
            V{i} = zeros(n(i),r(i));
        end
        
        eta.G = G;
        eta.V = V;
    end

    % Efficient vector transport by orthogonal projection, see Kressner at al.,
    % 2014, for idea
    M.transp = @transport;
    function eta = transport(X,Y,xi)

        % Take notation from Kressner et al. paper
        C = X.X.core; U = X.X.U; uList = U;
        U_tilde = Y.X.U; uTildeList = U_tilde;
        G = xi.G; V = xi.V;

        G_tilde = ttm(ttm(G,U),U_tilde,'t');

        for i = 1:d

            % List of basis matrices U without the index i
            uListWoI = U;
            uListWoI{i} = V{i};

            G_tilde = G_tilde + ttm(ttm(C,uListWoI),U_tilde,'t');
        end

        V_tilde = cell(1,d);
        for i = 1:d

            % Modes vector witout ith mode for ttm multiplication
            modesWoI = 1:d;
            modesWoI(i) = [];

            % List of basis matrices U without the index i
            uTildeListWoI = uTildeList;
            uTildeListWoI(:,i) = [];

            beforeProj = ttm(ttm(G,U),uTildeListWoI,modesWoI,'t');

            for k = 1:d
                uListWoK = uList;
                uListWoK{k} = V{k};

                beforeProj = beforeProj +...
                    ttm(ttm(C,uListWoK),uTildeListWoI,modesWoI,'t');
            end

            beforeProj = tenmat(beforeProj,i) * Y.Cpinv{i};

            V_tilde{i} = double(beforeProj - U_tilde{i}*(U_tilde{i}'*beforeProj));
        end

        eta.G = G_tilde;
        eta.V = V_tilde;
    end
    
    M.vec = @(X, eta) [eta.V{1}(:); eta.V{2}(:); eta.V{3}(:);eta.G(:)];

    M.mat = @normrep;
    function eta = normrep(X, eta_vec)
        
        V = cell(1,d);
        first_ind = 1;
        for i=1:d
            V{i} = reshape(eta_vec(first_ind : first_ind + n(i)*r(i)-1), n(i), r(i));
            first_ind = first_ind + n(i)*r(i);
        end
        G = tensor(reshape(eta_vec(first_ind : end), r));
        
        eta.G = G;
        eta.V = V;
    end

    % vec and mat are not isometries
    M.vecmatareisometries = @() false;
    
end

% Higher-order SVD, see De Lathauwer et al., 2000
function T = hosvd(X, r)
    if (ndims(X) == length(r))
        d = ndims(X);
    else
        error('Dimensions of tensor and multilinear rank vector do not match.')
    end

    % store matrices U of r left singular vectors of X and store ina cell array
    uList = cell(1,d);
    for i=1:d
        % convert do double because svds of tenmat impossible
        A = double(tenmat(X,i));
        [U,~,~] = svds(A,r(i));
        uList{i} = U;
    end

    C = ttm(X,uList,'t');

    T = ttensor(C,uList);
end

% Curvature term for Riemannian Hessian, see Heidel/Schulz, 2017, Corollary 3.7
function eta = curvature_term(E, X, xi)
    G = tenzeros(size(X.X.core));
    d = length(size(X.X.core));
    V = cell(1,d);

    for i=1:d
        modesWoI = 1:d;
        modesWoI(i) = [];

        uListWoI = X.X.U;
        uListWoI(:,i) = [];
        
        EUit = ttm(E,uListWoI,modesWoI,'t');
        Gi = double(tenmat(xi.G,i));
        Ci = double(tenmat(X.X.core,i));
        
        G = G + ttm(EUit,xi.V{i},i,'t')...
            - ttm(X.X.core,double(xi.V{i}'*(tenmat(EUit,i)*X.Cpinv{i})),i); 
        
        Cplusi2 = X.Cpinv{i}'*X.Cpinv{i};
        Vi = (tenmat(EUit,i)*Gi')*Cplusi2 + (tenmat(EUit,i)*X.Cpinv{i})*(Ci*Gi')*Cplusi2;
        for k = 1:length(modesWoI)
            modesWoIWoK = modesWoI;
            modesWoIWoK(k) = [];
            
            uListWoIWoK = uListWoI;
            uListWoIWoK(:,k) = [];

            EUiEUkdott = ttm(ttm(E,uListWoIWoK,modesWoIWoK,'t'),xi.V{modesWoI(k)},modesWoI(k),'t');
            Vi = Vi + tenmat(EUiEUkdott,i)*X.Cpinv{i};
        end
        V{i} = double(Vi - X.X.U{i}*(X.X.U{i}'*Vi));
    end

    eta.G = G;
    eta.V = V;
end
