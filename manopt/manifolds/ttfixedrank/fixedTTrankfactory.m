function M = fixedTTrankfactory(n, r, ind)
% Manifold of tensors of fixed Tensor Train (TT) rank, embedded geometry
% 
% NOTE: this manifold requires the use of a modified version of TTeMPS_1.1,
% which is packaced with Manopt and can be found in 
% /manopt/manifolds/ttfixedrank/TTeMPS_1.1/
% 
% function M = fixedTTrankfactory(n, r)
% function M = fixedTTrankfactory(n, r, ind)
% 
% Inputs:
%   n is a vector denoting the embedding space dimension,
%     R^{n(1) x ... x n(d)} (d = length(n) is the order of the tensor)
%   r is a vector denoting the TT-rank of all tensors in the manifold,
%     where length(r) = d + 1. We also enforce r(1) = r(d+1) = 1.
%   ind (optional): if only sparse tensors in the embedding space are
%     considered (as is the case for tensor completion in particular), the
%     parameter ind can be passed, where ind is a matrix of size p-by-d
%     whose rows contain the multi-indices of the p non-zero entries.
%     See TTeMPS_1.1/algorithms/completion/makeOmegaSet.m for an example
%     on constructing ind.
% 
% A point X on the manifold is represented through its TT-cores, stored in
% the cell array X.U. We enforce the TT-cores in X.U to be
% 'left-orthogonalized' (see Steinlechner's thesis, Section 4.2.1), because
% many algorithms require X.U to be left-orthogonalized.
% 
% A tangent vector Z in the tangent space of a TT-tensor X is represented
% as a structure containing 3 cell-arrays:
% 
% 1) Z.U, which is exactly X.U of the base point X
% 2) Z.V, the right-orthogonalization of X.U
% 3) Z.dU, the 'variational cores' that parametrize the tangent vector Z
%          itself. This matches the 'alternative representation' of tangent
%          vectors discussed in the Psenka and Boumal paper (see below).
% 
% The first-order Riemannian geometry of the manifold of fixed TT-rank
% tensors is described in detail in Steinlechner's PhD thesis, Section 4:
%   https://infoscience.epfl.ch/record/217938?ln=en
% TTeMPS also comes from that work.
%
% The second-order Riemannian geometry (necessary for the ehess2rhess tool)
% is described in the following paper:
%   Psenka and Boumal,
%   Second-order optimization for tensors with fixed tensor-train rank,
%   Optimization workshop at NeurIPS 2020.
%   https://arxiv.org/abs/2011.13395
% 
% Please cite the Manopt paper as well as the relevant research papers.
% 
% See also: fixedrankembeddedfactory fixedranktensorembeddedfactory

% This file is part of Manopt: www.manopt.org.
% Original author: Michael Psenka, Nov. 24, 2020.
% Contributors: Nicolas Boumal
% Change log: 

    % Order of tensors
    d = numel(n);
    
    assert(length(r) == d+1, ...
                    'Vector r must have length equal to length(n)+1.');
    assert(r(1) == 1 && r(end) == 1, ...
                    'The first and last entry of r must be equal to 1.');

    M.name = @() sprintf('Manifold of tensor order %d, dimension %s, and TT-rank %s', ...
        d, mat2str(n), mat2str(r));

    M.dim = @dim;
    function k = dim()
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
        X = TTeMPS_randn(r, n);
        % Left-orthogonalize X
        X = orthogonalize(X, d);
        % Normalize (efficient transformation to unit norm)
        X.U{d} = (1 / norm(X.U{d})) * X.U{d};
    end

    % Uses the TT-SVD algorithm to project a full array to a TT-tensor
    % of TT-rank r
    M.from_array = @fromArray;

    function X = fromArray(Y)
        % TTeMPS implementation of TT-SVD
        X = TTeMPS.from_array(Y, r);
        % Left-orthogonalized version of X
        X = orthogonalize(X, d);
    end

    % Creates random unit norm tangent vector at X on manifold.
    % Optional argument Xr of right-orthogonalized X.
    M.randvec = @randomTangent;
    function Z = randomTangent(X, Xr)
        if nargin == 1
            % If not provided, right orthogonalize X
            Xr = orthogonalize(X, 1); % right-orthogonalized version of X
        end

        % Two arguments --> random unit-norm tangent vector
        Z = TTeMPS_tangent_orth(X, Xr);
    end

    M.zerovec = @zeroVector;
    function Z0 = zeroVector(X)
        % X could be given as base point or tangent vector
        if isa(X, 'TTeMPS')
            % For TTeMPS function, one argument --> zero vec at point
            Z0 = TTeMPS_tangent_orth(X); 
        elseif isa(X, 'TTeMPS_tangent_orth')
            % If tangent vec, simply set dU cores to 0
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
    M.norm = @(x, v) real(sqrt(innerprod(v, v)));
    M.dist = @(x, y) error('tensor_fixed_TT_rank_factory.dist not implemented yet.');
    M.typicaldist = @() M.dim();


    % Given Z in tangent vector format, projects the components U_i such
    % that they satisfy the tangent space constraints up to numerical
    % errors (i.e., enforce that they satisfy the so-called gauge
    % conditions). If Z was indeed a tangent vector at X, this should
    % barely affect Z (it would not at all if we had infinite numerical
    % accuracy).
    M.tangent = @tangent;
    function Z = tangent(X, Zin)
        % Project to normal spaces of U^L for all but the last core
        Z = Zin; % this copies the TTeMPS_tangent_orth class structure
        for k = 1:(d-1)
            dUL = unfold(Zin.dU{k}, 'left');
            UL = unfold(X.U{k}, 'left');
            dUL_new = dUL - UL * (UL' * dUL);
            r = Z.rank;
            n = Z.size;
            Z.dU{k} = reshape(dUL_new, [r(k), n(k), r(k + 1)]);
        end
    end


    % It would be useful to implement the following efficienctly.
    % 
    % Applies a linear transformation to tensor W.
    % Z is a matrix, and W is a tensor, which must be flattened into a
    % vector before applying Z.
    % function ZW = apply_matrix(Z, W)
    %     ...
    % end
    % 
    % Same as apply_ambient, but applies Z' to W.
    % function ZtW = apply_matrix_transpose(Z, W)
    %     ...
    % end

    % Compute linear combination of two TT-tensors.
    % Note that '+' and scalar multiplication are both overloaded in the
    % TTeMPS library.
    M.lincomb = @matrixlincomb;


    % Orthogonal projection of an ambient vector Z represented as full
    % array of size n to the tangent space of TT-tensor Z.
    % Two possible calls: either xR (right orthogonalized) is known or not.
    M.proj = @projection;
    function Zproj = projection(X, Z)

        Xr = orthogonalize(X, 1); %right-orthogonalized version of X

        % Check if using sparse ambient space
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

    % Converts a tangent vector to the TT format
    M.tangent2TT = @tangent2TT;
    function Z_TT = tangent2TT(X, Z) %#ok<INUSL>
        Z_TT = tangent_to_TTeMPS(Z);
    end

    % It would be useful to implement the following more efficiently...
    M.tangent2ambient = @tangent2ambient;
    function Zamb = tangent2ambient(X, Z) %#ok<INUSL>
        Zamb = full(Z);
    end

    % Retraction for the fixed TT - rank manifold
    M.retr = @retraction;

    % NOTE: X not used in the function because Z is
    % a structure that contains all information of
    % X. X is kept as an argument to keep consistency
    % with standard manifold factory format
    function Y = retraction(X, Z, t) %#ok<INUSL>

        if nargin < 3
            t = 1;
        end

        Y = tangentAdd(Z, t, true);
        Y = orthogonalize(Y, d);

    end

    % Vector transport (see Steinlechner's thesis)
    % Computes a tangent vector at X2 that "looks like" the tangent vector
    % Z1 at X1. This is not necessarily a parallel transport.
    M.transp = @project_tangent;
    function Z2 = project_tangent(X1, X2, Z1) %#ok<INUSL>
        Z2 = projection(X2, Z1);
    end

    % The function 'vec' is isometric from the tangent space at X to real
    % vectors by flattening dU cores and stringing out to one vector.
    % The function 'mat' is the left-inverse of 'vec'. It is sometimes
    % useful to apply 'tangent' to the output of 'mat'.
    M.vec = @vec;
    function Zvec = vec(X, Z) %#ok<INUSL>
        X_size = 0;
        for k = 1:d
            X_size = X_size + numel(Z.dU{k});
        end
        % initialize full flattened vector in memory
        Zvec = zeros(X_size,1);
        % starting index to fill Zvec
        ind_start = 1;
        % how many entries to fill Zvec
        ind_step = numel(Z.dU{1});
        for k = 1:d
            Zvec(ind_start:ind_start+ind_step-1) = Z.dU{k}(:);
            ind_start = ind_start+ind_step;
            if k < d % avoids indexing error at end of loop
                ind_step = numel(Z.dU{k+1});
            end
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

    % That vec/mat are isometries is checked in the Psenka & Boumal paper,
    % in relation to the discussion of an alternative parametrization of
    % the tangent spaces.
    M.vecmatareisometries = @() true;

end
