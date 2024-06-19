function M = euclideanlargefactory(m, n)
% Returns a manifold struct to store and manipulate large real matrices.
%
% This factory outputs a structure in the Manopt format to store and
% manipulate elements of the linear space R^(m x n), with the trace metric.
%
% A point X (equivalently, a tangent vector) in this space may be
% represented in any of the following ways:
%
%   * As the matrix X itself (full or sparse, though preferably sparse)
%   * As a struct S with fields L and R, so that X = S.L * S.R.'
%   * As a struct Z with fields U, S and V, so that X = Z.U * Z.S * Z.V.'
%   * As a struct S with fields times and transpose_times so that
%       - S.times(A) = X*A for all A of size n x k (any k)
%       - S.transpose_times(B) = X.'*B for all B of size m x k (any k)
%
% These flexible formats make it possible to exploit structure such as
% sparsity and low rank (and mixtures of both) in order to store and
% operate on large matrices. This factory can serve as a useful description
% of the embedding space for, e.g., low-rank manifolds.
%
% Look inside the code for a list of functions made available.
%
% See also: euclideanfactory euclideansparsefactory
%           fixedrankembeddedfactory desingularizationfactory

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, June 18, 2024.
% Contributors: 
% Change log: 

    name = 'Euclidean space of large matrices of size %d x %d';
    M.name = @() sprintf(name, m, n);

    %% Helpers to determine the representation format of an input X.
    M.is_matrix = @is_matrix;
    M.is_LR     = @is_LR;
    M.is_USV    = @is_USV;
    M.is_funs   = @is_funs;

    %% Helpers to convert from any format to any other format.
    M.to_matrix = @to_matrix;
    M.to_USV    = @to_USV;
    M.to_LR     = @to_LR;
    M.to_funs   = @to_funs;

    %% Functions related to the linear space structure.
    M.zero      = @() spalloc(m, n, 1); % 0 as a sparse matrix of size mxn
    M.add       = @add;         % add(X, Y) = X+Y
    M.scale     = @scale;       % scale(a, X) = a*X
    M.diff      = @diff;        % diff(X, Y) = X-Y
    M.lincomb   = @lincomb;     % lincomb(X, a, U, b, V) = a*U + b*V
                                % X, b and V can be omitted.

    %% Functions to multiply X with another matrix, possibly a sparse one.
    M.times           = @times;             % times(X, A) = X*A
    M.transpose_times = @transpose_times;   % transpose_times(X, B) = X.'*B
    M.sparseentries   = @thissparseentries; % sample X from a sparse mask
    M.entrywisetimes  = @entrywisetimes;    % entries of X times a mask

    %% Functions related to the Euclidean manifold structure.
    M.dim             = @() m*n;
    M.proj            = @(X, U) U;
    M.tangent         = M.proj;
    M.retr            = @retraction;
    M.exp             = M.retr;
    M.transp          = @(X, Y, U) U;
    M.isotransp       = M.transp;
    M.tangent2ambient = @(X, U) U;
    M.zerovec         = @(X) spalloc(m, n, 1);
    M.rand            = @random;
    M.randvec         = @(X) random();
    M.egrad2rgrad     = M.proj;
    M.ehess2rhess     = @(X, egrad, ehess, U) ehess;
    M.inner           = @inner;   % inner(U, V) = <U, V> = trace(U'*V)
    M.norm            = @nrm;     % norm(U) = norm(U, 'fro')


    %% Helpers to determine the representation format of a point/vector X.

    % Matlab's builtin ismatrix(struct()) returns true somehow?
    is_matrix = @(X) isnumeric(X) && all(size(X) == [m, n]);
    is_LR = @(X) isstruct(X) && isfield(X, 'L') && isfield(X, 'R');
    is_USV = @(X) isstruct(X) && isfield(X, 'U') ...
                              &&  isfield(X, 'S') && isfield(X, 'V');
    is_funs = @(X) isstruct(X) && isfield(X, 'times') ...
                               && isfield(X, 'transpose_times');

    %% Helpers to convert from any format to a chosen format.

    function Y = to_matrix(X)  % This is expensive if m*n is large
        if is_matrix(X)
            Y = X;
        elseif is_LR(X)
            Y = X.L * X.R.';
        elseif is_funs(X)
            Y = X.times(eye(n));
        elseif is_USV(X)
            Y = X.U * X.S * X.V.';
        else
            error('Wrong format for X');
        end
    end
    function Y = to_USV(X)
        if is_matrix(X)
            [U, S, V] = svd(X);
            Y.U = U;
            Y.S = S;
            Y.V = V;
        elseif is_LR(X)
            [QL, RL] = qr(X.L, 0);
            [QR, RR] = qr(X.R, 0);
            [u, s, v] = svd(RL*RR.');
            Y.U = QL*u;
            Y.S = s;
            Y.V = QR*v;
        elseif is_funs(X)
            Y = to_USV(to_matrix(X));
        elseif is_USV(X)
            Y = X;
        else
            error('Wrong format for X');
        end
    end
    function Y = to_LR(X)
        if is_matrix(X)
            Y = to_LR(to_USV(X));
        elseif is_LR(X)
            Y = X;
        elseif is_funs(X)
            Y = to_LR(to_matrix(X));
        elseif is_USV(X)
            [u, s, v] = svd(X.S);
            Y.L = X.U*u*diag(sqrt(diag(s)));
            Y.R = X.V*v*diag(sqrt(diag(s)));
        else
            error('Wrong format for X');
        end
    end
    function Y = to_funs(X)
        if is_funs(X)
            Y = X;
        elseif is_matrix(X) || is_LR(X) || is_USV(X)
            Y.times = @(A) times(X, A);
            Y.transpose_times = @(B) transpose_times(X, B);
        else
            error('Wrong format for X');
        end
    end

    %% Functions related to the linear space structure.

    % Produce a representation Z for X+Y
    function Z = add(X, Y)
        if is_matrix(X)
            if is_matrix(Y)
                Z = X+Y;
            elseif is_LR(Y)
                Z.times = @(A) X*A + Y.L*(Y.R'*A);
                Z.transpose_times = @(B) X.'*B + Y.R*(Y.L.'*B);
            elseif is_funs(Y)
                Z.times = @(A) X*A + Y.times(A);
                Z.transpose_times = @(B) X.'*B + Y.transpose_times(B);
            else
                error('Wrong format for Y');
            end
        elseif is_LR(X)
            if is_matrix(Y)
                Z = add(Y, X);
            elseif is_LR(Y)
                Z.L = [X.L, Y.L];
                Z.R = [X.R, Y.R]; 
            elseif is_funs(Y)
                Z.times = @(A) X.L*(X.R.'*A) + Y.times(A);
                Z.transpose_times = @(B) X.R*(X.L.'*B) + ...
                                         Y.transpose_times(B);
            else
                error('Wrong format for Y');
            end
        elseif is_funs(X)
            if is_matrix(Y)
                Z = add(Y, X);
            elseif is_LR(Y)
                Z = add(Y, X);
            elseif is_funs(Y)
                Z.times = @(A) X.times(A) + Y.times(A);
                Z.transpose_times = @(B) X.transpose_times(B) + ...
                                         Y.transpose_times(B);
            else
                error('Wrong format for Y');
            end
        elseif is_USV(X)
            % If X is USV, make it LR and swap the two inputs.
            % If Y is not USV, we're fine.
            % If Y is also USV, it will be converted to LR in the
            % next call, swapped again, and now both are in LR.
            X_as_LR.L = X.U*X.S;
            X_as_LR.R = X.V;
            Z = add(Y, X_as_LR);
        else
            error('Wrong format for X');
        end
    end

    % Produce a representation Y for a*X
    function Y = scale(a, X)
        if is_matrix(X)
            Y = a*X;
        elseif is_LR(X)
            Y.L = sign(a)*sqrt(abs(a))*X.L;
            Y.R =         sqrt(abs(a))*X.R;
        elseif is_USV(X)
            Y.U = X.U;
            Y.S = a*X.S;
            Y.V = X.V;
        elseif is_funs(X)
            Y.times = @(A) a*X.times(A);
            Y.transpose_times = @(B) a*X.transpose_times(B);
        else
            error('Wrong format for X');
        end
    end

    % Produce a representation Z for X-Y
    function Z = diff(X, Y)
        Z = add(X, scale(-1, Y));
    end

    % Produce a representation W for a*U + b*V (X, b, V optional)
    function W = lincomb(X, a, U, b, V)
        switch nargin
            case 2 % (a, U) -> W = a*U  (X omitted)
                W = lincomb([], X, a);
            case 3 % (X, a, U) -> W = a*U
                W = scale(a, U);
            case 4 % (a, U, b, V) -> W = a*U + b*V  (X omitted)
                W = lincomb([], X, a, U, b);
            case 5 % (X, a, U, b, V) -> W = a*U + b*V
                aU = scale(a, U);
                bV = scale(b, V);
                W = add(aU, bV);
            otherwise
                error('lincomb takes 2, 3, 4 or 5 inputs.');
        end
    end

    %% Functions to multiply X with another matrix, possibly a sparse one.

    % Compute the product C = X*A
    function C = times(X, A)
        if is_matrix(X)
            C = X*A;
        elseif is_LR(X)
            C = X.L*(X.R.'*A);
        elseif is_funs(X)
            C = X.times(A);
        elseif is_USV(X)
            C = X.U*(X.S*(X.V.'*A));
        else
            error('Wrong format for X');
        end
    end

    % Compute the product C = X.'*B
    function C = transpose_times(X, B)
        if is_matrix(X)
            C = X.'*B;
        elseif is_LR(X)
            C = X.R*(X.L.'*B);
        elseif is_funs(X)
            C = X.transpose_times(B);
        elseif is_USV(X)
            C = X.V*(X.S.'*(X.U.'*B));
        else
            error('Wrong format for X');
        end
    end

    % Given a sparse matrix mask and a point X,
    % computes the entries of X corresponding to the sparsity pattern of
    % the mask, as a vector in the order corresponding to find(mask).
    function x = thissparseentries(mask, X)
        if is_matrix(X)
            assert(all(size(mask) == size(X)), ...
                   'X and the mask must have same size.');
            ij = find(mask);
            x = X(ij);
        elseif is_LR(X)
            x = sparseentries(mask, X.L, X.R);
        elseif is_USV(X)
            x = sparseentries(mask, X.U*X.S, X.V);
        elseif is_funs(X)
            % In principle, this could be improved.
            % One option would be to add a function field X.sample()
            % or X.entries() as part of the functions description of X.
            x = thissparseentries(mask, as_matrix(X));
        else
            error('Wrong format for X');
        end
    end

    % Same as M.sparseentries but the computed entries of X are entry-wise
    % multiplied with their matching entry in sparse_matrix.
    function x = entrywisetimes(sparse_matrix, X)
        if is_matrix(X)
            assert(all(size(sparse_matrix) == size(X)), ...
                   'X and the sparse matrix must have same size.');
            [I, J, Mvals] = find(sparse_matrix);
            x = Mvals .* X(sub2ind(size(sparse_matrix), I, J));
        elseif is_LR(X)
            x = sparseentrywisemult(sparse_matrix, X.L, X.R);
        elseif is_USV(X)
            x = sparseentrywisemult(sparse_matrix, X.L*X.S, X.R);
        elseif is_funs(X)
            % In principle, this could be improved.
            % One option would be to add a function field X.sample()
            % or X.entries() as part of the functions description of X.
            x = entrywisetimes(mask, as_matrix(X));
        else
            error('Wrong format for X');
        end
    end


    %% Euclidean structure: trace inner product and Frobenius norm

    inr = @(A, B) A(:).'*B(:);

    function val = inner(X, U, V)
        % X is not necessary for a Euclidean manifold, so we allow calling
        % this function as inner(U, V), omitting X.
        if nargin == 2
            val = inner([], X, U);
            return;
        end
        % Convert any USV format to LR format.
        if is_USV(U)
            val = inner(X, to_LR(U), V);
            return;
        end
        if is_USV(V)
            val = inner(X, U, to_LR(V));
            return;
        end
        % If either U or V is in LR format, use that first.
        if is_LR(U)
            val = inr(U.L, times(V, U.R));
            return;
        end
        if is_LR(V)
            val = inr(times(U, V.R), V.L);
            return;
        end
        % We now know that neither U nor V are in LR or USV format.
        % If both are in matrix format (sparse or not), a direct
        % computation is likely the most efficient.
        if is_matrix(U) && is_matrix(V)
            val = sum(U.*V, 'all');
            return;
        end
        % We now know that U or V is in functions format, and that the
        % other is either also in functions format, or it is a matrix.
        if is_funs(U)
            if is_matrix(V)
                val = trace(U.transpose_times(V));             % slow
            elseif is_funs(V)
                val = trace(U.transpose_times(to_matrix(V)));  % very slow
            else
                error('Wrong format');
            end
            return;
        elseif is_funs(V)
            val = inner(X, V, U);
            return;
        end
    end

    function val = nrm(X, U)
        switch nargin
            case 1
                val = nrm([], X);
            case 2
                if is_matrix(U)
                    val = norm(U, 'fro');
                elseif is_LR(U)
                    val = sqrt(inr(U.R.'*U.R, U.L.'*U.L));
                elseif is_USV(U)
                    % Could be faster if we assume U, V are orthonormal.
                    val = nrm(X, struct('L', U.U*U.S, 'R', U.V));
                elseif is_funs(U)
                    val = norm(to_matrix(U), 'fro');
                else
                    error('Wrong format for U');
                end
            otherwise
                error('norm takes 1 or 2 inputs.');
        end
    end

    %% Functions related to the linear manifold structure.

    function Y = retraction(X, U, t)
        if nargin == 2
            Y = add(X, U); % t = 1 by default
        else
            Y = add(X, scale(t, U));
        end
    end

    % There is no good default choice of a random large matrix.
    % The code below arbitrarily generates a random matrix with random
    % rank between 1 and 20.
    function X = random()
        r = randi(20);
        X.L = randn(m, r);
        X.R = randn(n, r);
        X = scale(1/nrm(X), X);
    end

end
