
function M = fixedTTrankfactory(n, r, ind)

% Manifold struct to optimize over tensors of fixed Tensor Train (TT) rank
% with an embedded geometry.

% EXTERNAL REQUIREMENT: TTeMPS_1.1, which can be found here:
% https://www.epfl.ch/labs/anchp/index-html/software/ttemps/

% Constructor has two overflow calls:

% function M = fixedTTrankfactory(n, r)

% n is a vector denoting the embedding space dimension (thus d=size(n) is the
% order of the tensor), and r is a vector denoting the TT-rank of all tensors
% in the manifold, where size(r) = d + 1. We also enforce r(1) = r(d+1) = 1

% function M = fixedTTrankfactory(n, r, ind)

% If only sparse tensors in the embedding space are considered (e.g. tensor
% completion), the parameter ind can be passed, where ind is a (p,size(n)) size
% matrix denoting the set of multi-indices of non-zero entries, of which there are k.
% See TTeMPS_1.1/algorithms/completion/makeOmegaSet.m for an example on constructing
% ind.

% Main paper link for Manopt Tensor Train library: (link to arXiv).

% Steinlechner's thesis Section 4 contains details on the manifold of tensors
% with fixed Tensor Train rank: https://infoscience.epfl.ch/record/217938?ln=en

% Points X in the manifold are represented by the TT-cores, which is given by the
% cell array X.U. We enforce the TT-cores in X.U to be 'left-orthogonalized' (see
% Steinlechner's thesis, Section 4.2.1), as many algorithms require X.U to be
% left-orthogonalized.

% Tangent vectors Z in the tangent space of a TT-tensor X are represented as a
% structure of 3 cell-arrays:

% 1) Z.U, which is exactly X.U of the base point X
% 2) Z.V, the right-orthogonalization of X.U
% 3) Z.dU, the 'variational cores' that parametrize the tangent vector Z itself.
%          This is given by the 'alternative representation' in Section 3 of
%          the main paper.


% Please cite the Manopt paper as well as the research paper:
%     @Article{vandereycken2013lowrank,
%       Title   = {Low-rank matrix completion by {Riemannian} optimization},
%       Author  = {Vandereycken, B.},
%       Journal = {SIAM Journal on Optimization},
%       Year    = {2013},
%       Number  = {2},
%       Pages   = {1214--1236},
%       Volume  = {23},
%       Doi     = {10.1137/110845768}
%     }

% This file is part of Manopt: www.manopt.org

% Author of fixedTTrankfactory.m: Michael Psenka
    % order of tensors
    d = numel(n);

    M.name = @() sprintf('Manifold of tensor order %d, dimension %s, and TT-rank %s', ...
        d, mat2str(n), mat2str(r));

    M.dim = @dimVal;

    function k = dimVal()
        k = 0;

        for m = 1:(d-1)
            k = k + r(m) * n(m) * r(m + 1) - r(m + 1) * r(m + 1);
        end

        k = k + r(d) * n(d) * r(d + 1);
    end

    % Creates random unit-norm TT-tensor of TT-rank r.
    M.rand = @random;

    function X = random()
        % Gaussian random cores
        X = TTeMPS_rand(r, n);
        % left-orthogonalize X
        X = orthogonalize(X, d);
        % normalize
        X.U{d} = (1 / norm(X.U{d})) * X.U{d}; % efficient transformation to unit norm
    end

    % Uses the TT-SVD algorithm to project a full array to a TT-tensor of TT-rank r
    M.from_array = @fromArray;

    function X = fromArray(Y)
        % TTeMPS implementation of TT-SVD
        X = TTeMPS.from_array(Y, r);
        %left-orthogonalized version of X
        X = orthogonalize(X, d);
    end

    % Creates random unit norm tangent vector at X on manifold
    M.randvec = @randomTangent;
    % TODO name randvec and use randn

    % optional argument Xr of right-orthogonalized X
    function Z = randomTangent(X, Xr)
        if nargin == 1
            % If not provided, right orthogonalize X
            Xr = orthogonalize(X, 1); %right-orthogonalized version of X
        end

        Z = TTeMPS_tangent_orth(X, Xr); % two arguments --> random unit norm tangent
    end

    M.zerovec = @zeroVector;

    function Z0 = zeroVector(X)
        % X could be given as base point or tangent vector
        if isa(X, 'TTeMPS')
            % for TTeMPS function, one argument --> zero vec at point
            Z0 = TTeMPS_tangent_orth(X); 
        elseif isa(X, 'TTeMPS_tangent_orth')
            % if tangent vec, simply set dU cores to 0
            Z0 = X;
            for k = 1:d
                Z0.dU{k} = zeros(r(k), n(k), r(k + 1));
            end
        else
            error('unexpected input type for zerovec')
        end

    end

    % Note that innerprod has an overflow in TTeMPS for TT-tensor arguments
    M.inner = @(x, u, v) innerprod(u, v);
    M.norm = @(x, v) sqrt(innerprod(v, v));
    M.dist = @(x, y) error('tensor_fixed_TT_rank_factory.dist not implemented yet.');
    M.typicaldist = @() M.dim();


    % Given Z in tangent vector format, projects the components U_i
    % such that they satisfy the tangent space constraints up to numerical
    % errors (i.e. enforce that they satisfy the so-called 'gauge conditions').
    % If Z was indeed a tangent vector at X, this should barely
    % affect Z (it would not at all if we had infinite numerical accuracy).

    M.tangent = @tangent;

    function Z = tangent(X, Zin, Xr)
        % Here, we project to normal spaces of U^L for all cores except for the last core
        Z = Zin; % merely used to get TTeMPS_tangent_orth class structure
        for k = 1:(d-1)
            dUL = unfold(Zin.dU{k}, 'left');
            UL = unfold(X.U{k}, 'left');
            dUL_new = dUL - UL * (UL' * dUL);
            r = Z.rank;
            n = Z.size;
            Z.dU{k} = reshape(dUL_new, [r(k), n(k), r(k + 1)]);
        end
    end


    % Applies a linear transformation to tensor W. Z is a matrix, and W is
    % a tensor, which must be flattened into a vector before applying Z.

    function ZW = apply_matrix(Z, W)
        disp('TO-DO');
    end

    % Same as apply_ambient, but applies Z' to W.
    function ZtW = apply_matrix_transpose(Z, W)
        disp('TO-DO');
    end

    % Compute linear combination of two TT-tensors
    % Note that '+' and scalar multiplication both
    % have overflows in the TTeMPS library.
    M.lincomb = @lin_comb;

    function XY = lin_comb(x, a1, u1, a2, u2)

        if nargin == 3% only a1 and u1 specified
            XY = a1 * u1;

        elseif nargin == 4% a2 additional specified
            XY = a1 * u1 + aa2int;

        else
            XY = (a1 * u1) + (a2 * u2);
        end

    end


    % Orthogonal projection of an ambient vector Z represented as full
    % array of size n to the tangent space of TT-tensor Z.
    M.proj = @projection;

    % Two possible calls: either xR (right orthogonalized) is known or not.
    function Zproj = projection(X, Z)

        Xr = orthogonalize(X, 1); %right-orthogonalized version of X

        % Checks if using sparse ambient space
        if exist('ind', 'var')
            Zproj = TTeMPS_tangent_orth(X, Xr, Z, ind);
        else
            Zproj = TTeMPS_tangent_orth(X, Xr, Z);
        end

    end

    M.egrad2rgrad = @projection;


    % Given the Euclidean gradient at X and the Euclidean Hessian at X
    % along H, where egrad and ehess are vectors in the ambient space and H
    % is a tangent vector at X, returns the Riemannian Hessian at X along
    % H, which is a tangent vector.
    % Curvature part denotes the Weingarten map part. Euclidean part is the
    % Euclidean hessian projection part,

    M.ehess2rhess = @ehess2rhess;

    function rhess = ehess2rhess(X, egrad, ehess, H)

        % Euclidean part
        rhess = projection(X, ehess);

        % Curvature part
        if exist('ind', 'var')
            rhess = rhess + TT_weingarten(H, egrad, ind);
        else
            rhess = rhess + TT_weingarten(H, egrad);
        end
    end

    %{
    Converts a tangent vector to the TT format
    %}
    M.tangent2TT = @tangent2TT;

    function Z_TT = tangent2TT(X, Z)
        Z_TT = tangent_to_TTeMPS(Z);
    end

    % TODO: tangent2ambient

    % Retraction for the fixed TT - rank manifold
    M.retr = @retraction;

    function Y = retraction(X, Z, t)

        if nargin < 3
            t = 1;
        end

    
        Y = tangentAdd(Z, t, true);
        Y = orthogonalize(Y, d); % ?

    end

    % Vector transport(see Steinleichner thesis)
    M.transp = @project_tangent;


    % Computes a tangent vector at X2 that "looks like" the tangent vector Z1 at X1
    % This is not necessarily a parallel transport.
    function Z2 = project_tangent(X1, X2, Z1)
        Z2 = projection(X2, Z1);
    end

     % The function 'vec' is isometric from the tangent space at X to real
    % vectors by flattening dU cores and stringing out to one vector.
    % The function 'mat' is the left-inverse of 'vec'. It is sometimes
    % useful to apply 'tangent' to the output of 'mat'.
    M.vec = @vec;
    function Zvec = vec(X, Z)
        Zvec = Z.dU{1}(:);
        for k = 2:d
            Zvec = [Zvec; Z.dU{k}(:)];
        end
    end

    M.mat = @mat;
    function Zmat = mat(X, Zvec)
        Zmat = M.zerovec(X);
        for k = 1:d
            sizeSubVec = numel(Zmat.dU{k}); % how many elements from Zvec belong to dU{k}
            Zmat.dU{k} = reshape(Zvec(1:sizeSubVec), size(Zmat.dU{k}));
            Zvec = Zvec((sizeSubVec+1):end);
        end        

    end

    % This is true; see Section 3, 'An alternative parametrization...' of Main paper
    M.vecmatareisometries = @checkisometries;
   
    function bool = checkisometries()
        bool = true;
    end

end
