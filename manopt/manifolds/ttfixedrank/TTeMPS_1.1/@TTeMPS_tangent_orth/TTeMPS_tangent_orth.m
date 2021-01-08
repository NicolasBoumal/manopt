classdef TTeMPS_tangent_orth
    % TTeMPS_tangent
    %
    %   A MATLAB class for representing and tangent tensors
    %   to the TT/MPS format in the core-by-core orthogonalization
    %   presented in the paper
    %
    %   Michael Steinlechner. Riemannian optimization for high-dimensional tensor completion
    %   Technical report, March 2015. Revised December 2015. To appear in SIAM J. Sci. Comput.
    %

    %   TTeMPS Toolbox.
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    properties (SetAccess = public, GetAccess = public)

        dU
        U
        V
        rank
        order
        size

    end

    methods (Access = public)

        function Y = TTeMPS_tangent_orth(xL, xR, Z, ind, storedb, key)
            %TTEMPS_TANGENT projects the d-dimensional array Z into the tangent space at a TT/MPS tensor X.
            %
            %   P = TTEMPS_TANGENT(X) projects the d-dimensional array Z into the tangent space of the
            %   TT-rank-r manifold at a TT/MPS tensor X.
            %

            d = xL.order;
            r = xL.rank;
            n = xL.size;

            % additional conditional needed for added code for ManOpt
            if nargin == 1
                xR = orthogonalize(xL, 1);
            end

            Y.order = d;
            Y.rank = r;
            Y.size = n;

            Y.U = xL.U;

            Y.V = xR.U;

            %%%%%%%%%%%%%%%%%% NEW CODE FOR MANOPT: add constructor for zero vector %%%%%%%%%%%%%%%%%%%%
            % Takes the unused spot nargin == 1      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            if nargin == 1
                % Y.U = xL.U;
                Y.V = xR.U;
                Y.dU = cell(1, d);

                for k = 1:d
                    Y.dU{k} = zeros(size(Y.U{k}));
                end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % default constructor which just returns a random unit norm tangent vector
            elseif nargin == 2
                % Y.U = xL.U;
                % Y.V = xR.U;
                Y.dU = cell(1, d);

                for i = 1:d
                    Y.dU{i} = randn(size(Y.U{i}));
                end

                Y = TTeMPS_tangent_orth(xL, xR, Y);

                for i = 1:d
                    Y.dU{i} = Y.dU{i} / (sqrt(d) * norm(Y.dU{i}(:)));
                end

            else

                if ~exist('ind', 'var')
                    sampled = false;
                else
                    sampled = true;
                end

                if isa(Z, 'TTeMPS')

                    Y.dU = cell(1, d);

                    right = innerprod(xR, Z, 'RL', 2, true);
                    left = innerprod(xL, Z, 'LR', d - 1, true);

                    % contract to first core
                    Y.dU{1} = tensorprod(Z.U{1}, right{2}, 3);
                    % contract to inner cores
                    for i = 2:d - 1
                        res = tensorprod(Z.U{i}, left{i - 1}, 1);
                        Y.dU{i} = tensorprod(res, right{i + 1}, 3);
                    end

                    % contract to last core
                    Y.dU{d} = tensorprod(Z.U{d}, left{d - 1}, 1);

                    for i = 1:d - 1
                        Y.dU{i} = unfold(Y.dU{i}, 'left');
                        U = unfold(Y.U{i}, 'left');
                        Y.dU{i} = Y.dU{i} - U * (U' * Y.dU{i});
                        Y.dU{i} = reshape(Y.dU{i}, [r(i), n(i), r(i + 1)]);
                    end

                elseif isa(Z, 'TTeMPS_tangent_orth')

                    Znew = tangent_to_TTeMPS(Z);
                    Y = TTeMPS_tangent_orth(xL, xR, Znew);

                elseif ~sampled % Z is a full array

                    ZZ = cell(1, d);

                    % right side
                    ZZ{d} = Z(:);

                    for i = d - 1:-1:1
                        zz = reshape(Z, [prod(n(1:i)), n(i + 1) * r(i + 2)]);
                        xx = transpose(unfold(Y.V{i + 1}, 'right'));
                        Z = zz * xx;
                        ZZ{i} = Z;
                    end

                    % left side
                    for k = 2:d

                        for i = 1:k - 1
                            Z_i = reshape(ZZ{k}, [r(i) * prod(n(i)), prod(n(i + 1:k)) * r(k + 1)]);
                            X_i = unfold(Y.U{i}, 'left');
                            Z = X_i' * Z_i;
                            ZZ{k} = Z;
                        end

                        ZZ{k} = reshape(ZZ{k}, [r(k) * n(k), r(k + 1)]);
                    end

                    Y.dU = cell(1, d);
                    % orth. projection (w/o last core)
                    for i = 1:d - 1
                        U = unfold(Y.U{i}, 'left');
                        ZZ{i} = ZZ{i} - U * (U' * ZZ{i});
                        Y.dU{i} = reshape(ZZ{i}, [r(i), n(i), r(i + 1)]);
                    end

                    Y.dU{d} = reshape(ZZ{d}, [r(d), n(d), r(d + 1)]);

                else % Z is a sparse array

                    vals = Z;
                    CU = cell(1, d);
                    CV = cell(1, d);
                    Y.dU = cell(1, d);

                    for i = 1:d
                        CU{i} = permute(Y.U{i}, [1 3 2]);
                        CV{i} = permute(Y.V{i}, [1 3 2]);
                    end

                    res = TTeMPS_tangent_orth.TTeMPS_tangent_orth_omega(n, r, CU, CV, ind.', vals);

                    %%%%%%%%%%%%%%%%%%%%%%% MANOPT ADDED CODE %%%%%%%%%%%%%%%%%%%%%%
                    if exist('storedb', 'var')
                    % store res to be used for efficient Weingarten approx
                        store = storedb.getWithShared(key);
                        store.inner_dU_ = res;
                        storedb.setWithShared(store, key);
                    end

                    %%%%%%%%%%%%%%%%%%%%%%% END MANOPT ADDED CODE %%%%%%%%%%%%%%%%%%

                    for i = 1:d
                        res{i} = reshape(res{i}, [r(i), r(i + 1), n(i)]);
                        Y.dU{i} = unfold(permute(res{i}, [1 3 2]), 'left');
                    end

                    for i = 1:d - 1
                        U = unfold(Y.U{i}, 'left');
                        Y.dU{i} = Y.dU{i} - U * (U' * Y.dU{i});
                        Y.dU{i} = reshape(Y.dU{i}, [r(i), n(i), r(i + 1)]);
                    end

                    Y.dU{d} = reshape(Y.dU{d}, [r(d), n(d), r(d + 1)]);

                end

            end

        end

        function Zfull = full(Z)
            %FULL converts tangent tensor to d-dimensional array
            %
            %   ZFULL = full(Z, X) converts the tangent tensor Z given in factorized form
            %   (class TTeMPS_tangent) to a d-dimensional array ZFULL. X is the TTeMPS tensor at
            %   which point the tangent space is taken.
            %

            Zfull = tangent_to_TTeMPS(Z);
            Zfull = full(Zfull);
        end

        function res = plus(X, Y)
            %PLUS adds two tangent tensors
            %
            %   RES = plus(X, Y) adds two tangent tensors in factorized form. Both tangent tensors
            %   have be elements of the SAME tangent space.
            %

            res = X;
            res.dU = cellfun(@plus, X.dU, Y.dU, 'UniformOutput', false);
        end

        function X = minus(X, Y)
            %MINUS substracts two tangent tensors
            %
            %   RES = minus(X, Y) substracts two tangent tensors in factorized form. Both tangent tensors
            %   have be elements of the SAME tangent space.
            %

            X.dU = cellfun(@minus, X.dU, Y.dU, 'UniformOutput', false);
        end

        function X = mtimes(a, X)
            %MTIMES Multiplication of TTeMPS tangent tensor by scalar
            %
            %   RES = mtimes(a, X) multiplies the TTeMPS tangent tensor X
            %   by the scalar a.
            %

            X.dU = cellfun(@(x) a * x, X.dU, 'UniformOutput', false);
        end

        function X = uminus(X)
            %UMINUS Unary minus of TTeMPS tangent tensor.
            %
            %   RES = uminus(X) negates the TTeMPS tangent tensor X.
            %

            X = mtimes(-1, X);
        end

        function Xnew = tangentAdd(Z, t, retract)
            %TANGENTADD adds a tangent vector to a point on the manifold
            %
            %   RES = tangentAdd(Z, t ) adds a tangent vector Z to the current point on the rank-r-manifold, scaled by t:
            %           res = X + t*Z
            %   where the result is stored as a TTeMPS tensor of rank 2*r.
            %
            %   RES = tangentAdd(Z, t, true) adds a tangent vector Z to the current point X on the rank-r-manifold, scaled by t:
            %           res = X + t*Z
            %   and retracts the result back to the manifold:
            %           res = R_X( X + t*Z )
            %   where the result is stored as a right orthogonal TTeMPS tensor of rank r.
            %

            if ~exist('retract', 'var')
                retract = false;
            end

            d = length(Z.dU);
            r = ones(1, d + 1);
            C = cell(1, d);

            C{1} = cat(3, t * Z.dU{1}, Z.U{1});

            for i = 2:d - 1
                sz = size(Z.U{i});
                r(i) = sz(1);
                zeroblock = zeros(sz);
                tmp1 = cat(3, Z.V{i}, zeroblock);
                tmp2 = cat(3, t * Z.dU{i}, Z.U{i});
                C{i} = cat(1, tmp1, tmp2);
            end

            r(d) = size(Z.U{d}, 1);
            C{d} = cat(1, Z.V{d}, Z.U{d} + t * Z.dU{d});
            Xnew = TTeMPS(C);

            if retract
                Xnew = truncate(Xnew, r);
            end

        end

        function res = innerprod(Z1, Z2)

            % due to left-and right orth., inner prod is just the inner product of the dU
            res = 0;

            for i = 1:length(Z1.dU)
                res = res + Z1.dU{i}(:)' * Z2.dU{i}(:);
            end

        end

        function n = norm(Z)
            Z_tt = tangent_to_TTeMPS(Z);
            n = norm(Z_tt);
        end

        function res = at_Omega(Z, ind)

            Xnew = tangent_to_TTeMPS(Z);
            res = Xnew(ind);
        end

        function res = tangent_to_TTeMPS(Z)
            d = length(Z.dU);
            C = cell(1, d);

            C{1} = cat(3, Z.dU{1}, Z.U{1});

            for i = 2:d - 1
                zeroblock = zeros(size(Z.U{i}));
                tmp1 = cat(3, Z.V{i}, zeroblock);
                tmp2 = cat(3, Z.dU{i}, Z.U{i});
                C{i} = cat(1, tmp1, tmp2);
            end

            C{d} = cat(1, Z.V{d}, Z.dU{d});

            res = TTeMPS(C);
        end

        function z = vectorize_tangent(Z)
            z = cellfun(@(x) x(:), Z.dU, 'UniformOutput', false);
            z = cell2mat(z(:));
        end

        function Z = fill_with_vectorized(Z, z)
            d = length(Z.dU);
            k = 1;

            for i = 1:d
                s = size(Z.dU{i});
                Z.dU{i} = reshape(z(k:k + prod(s) - 1), s);
                k = k + prod(s);
            end

        end

        function xi = tangent_orth_to_tangent(Z)
            % Transform a TTeMPS_tangent_orth to a TTeMPS_tangent.
            d = Z.order;
            xL = TTeMPS(Z.U);
            [xL, xR, G] = gauge_matrices(xL);

            xi = TTeMPS_tangent(xL);

            for ii = 1:d - 1
                xi.dU{ii} = tensorprod(Z.dU{ii}, inv(G{ii}'), 3);
            end

            % the lc;last one does not need to be changed
            xi.dU{d} = Z.dU{d};
        end

    end

    methods (Static, Access = private)

        x = TTeMPS_tangent_orth_omega(n, r, CU, CV, ind, vals);
        x = TTeMPS_tangent_orth_omega_openmp(n, r, CU, CV, ind, vals);

    end

end
