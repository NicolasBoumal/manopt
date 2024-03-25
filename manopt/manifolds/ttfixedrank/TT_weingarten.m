function Y = TT_weingarten(V, Z, ind)
% Weingarten map computation for the fixed TT-rank manifold.
%
% NOTE: this manifold requires the use of a modified version of TTeMPS_1.1,
% which is packaged with Manopt and can be found in 
% manopt/manopt/manifolds/ttfixedrank/TTeMPS_1.1
% It also uses some MEX files that you may need to compile by
% running install_mex.m in manopt/manopt/manifolds/ttfixedrank.
% 
% Y = TT_weingarten(V, Z, ind)
% 
% This function implements the efficient way of computing the Weingarten
% map for the Riemannian submanifold of fixed TT-rank tensors embedded in
% its natural embedding Euclidean space, as described in the paper:
%
%   Psenka and Boumal,
%   Second-order optimization for tensors with fixed tensor-train rank,
%   Optimization workshop at NeurIPS 2020.
% 
% Y is the output tangent vector, V is a tangent vector (which contains
% needed information of base point X), Z is a tensor (embedding space).
% 
% TT_weingarten supports the following formats for Z:
% 1) Full
% 2) TT-format (any rank)
% 3) sparse, with non-zero indices indexed by ind (see fixedTTrankfactory).
%
% See also: fixedTTrankfactory

% This file is part of Manopt: www.manopt.org.
% Original author: Michael Psenka, Nov. 24, 2020.
% Contributors: Nicolas Boumal
% Change log:

    % Preliminary tangent vector set-up for Y
    d = V.order;
    r = V.rank;
    n = V.size;
    

    Y = V; % all properties except for dU cores inherited from V
    normV = norm(V);
    V = (1 / normV) * V;
    Y.dU = cell(1, d);



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Begin calculating variational components
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Pre-calculate all R-components as needed for tangent space conversion
    R = cell(1, d-1);
    % Intermediary used to calculate R-terms for re-orthogonalization
    Xr = V.U;

    for k = d:-1:2

        sz = size(Xr{k});

        if length(sz) == 2
            sz = [sz, 1]; %#ok<AGROW>
        end

        [Q, R{k - 1}] = qr_unique(unfold(Xr{k}, 'right')');
        Xr{k} = reshape(Q', [size(Q, 2), sz(2), sz(3)]);

        Xr{k - 1} = tensorprod_ttemps(Xr{k - 1}, R{k - 1}, 3);
    end

    % Calculate V.dU under the original parametrization of tangent space

    dUR = cell(1, d);

    for k = 1:(d - 1)
        dUR{k} = unfold(V.dU{k}, 'left') / R{k}';
        dUR{k} = reshape(dUR{k}, [r(k), n(k), r(k + 1)]);
    end

    dUR{d} = V.dU{d};

    % Calculate ~X^tV_{\ge k} efficiently
    XtVg = cell(1, d);

    XtVg{d} = conj(unfold(V.V{d}, 'right')) * unfold(V.dU{d}, 'right').';
    XtVgTmp = conj(unfold(V.V{d}, 'right')) * unfold(V.U{d}, 'right').';

    for k = (d - 1):-1:2
        tmp = tensorprod_ttemps(V.V{k}, XtVg{k + 1}', 3);
        XtVg{k} = conj(unfold(tmp, 'right')) * unfold(V.U{k}, 'right').';

        tmp2 = tensorprod_ttemps(V.V{k}, XtVgTmp', 3);
        XtVg{k} = XtVg{k} + conj(unfold(tmp2, 'right')) * unfold(dUR{k}, 'right').';
        XtVgTmp = conj(unfold(tmp2, 'right')) * unfold(V.U{k}, 'right').';
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% CASES FOR TYPE OF EUCLIDEAN GRADIENT Z %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if isa(Z, 'TTeMPS') || isa(Z, 'TTeMPS_tangent_orth')% Z either TT-tensor or tangent vec

        if isa(Z, 'TTeMPS_tangent_orth')
            Z = tangent_to_TTeMPS(Z); % efficient to treat both as TTeMPS case
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%               DIAGONAL TERMS                 %%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        ZVr = cell(1, d); % stores values of V_<^T(Z<k>)
        Zr = cell(1, d);  % stores values of X_<^T(Z<k>)
        ZVl = cell(1, d); % stores values of (Z<k>)^T(V_>)
        Zl = cell(1, d);  % stores values of (Z<k>)^T(X_>)


        ZVr{d}  = conj(unfold(Z.U{d}, 'right')) * unfold(V.dU{d}, 'right').';
        XtVgTmp = conj(unfold(Z.U{d}, 'right')) * unfold(V.U{d},  'right').';

        for k = (d - 1):-1:2
            tmp = tensorprod_ttemps(Z.U{k}, ZVr{k + 1}', 3);
            ZVr{k} = conj(unfold(tmp, 'right')) * unfold(V.U{k}, 'right').';

            tmp2 = tensorprod_ttemps(Z.U{k}, XtVgTmp', 3);
            ZVr{k} = ZVr{k} + conj(unfold(tmp2, 'right')) * unfold(dUR{k}, 'right').';
            XtVgTmp = conj(unfold(tmp2, 'right')) * unfold(V.U{k}, 'right').';
        end

        for k=1:d
            ZVr{k} = ZVr{k}';
        end
   

        Zr{d} = conj(unfold(V.V{d}, 'right')) * unfold(Z.U{d}, 'right').';

        for k = (d - 1):-1:2
            tmp = tensorprod_ttemps(V.V{k}, Zr{k + 1}', 3);
            Zr{k} = conj(unfold(tmp, 'right')) * unfold(Z.U{k}, 'right').';
        end

        % Computation of ZVl and Zl

        ZVl{1} = unfold(dUR{1}, 'left')' * unfold(Z.U{1}, 'left');
        Zl{1}  = unfold(V.U{1}, 'left')' * unfold(Z.U{1}, 'left');

        for k = 2:(d - 1)
            % (V_k-1)(U_k)
            tmp = tensorprod_ttemps(V.U{k}, ZVl{k - 1}', 1);
            ZVl{k} = unfold(tmp, 'left')' * unfold(Z.U{k}, 'left');
            % + (X_k-1)(dU_k)
            tmp = tensorprod_ttemps(dUR{k}, Zl{k - 1}', 1);
            ZVl{k} = ZVl{k} + unfold(tmp, 'left')' * unfold(Z.U{k}, 'left');
            % update Zr to keep up w/ recursive definition
            tmp = tensorprod_ttemps(V.U{k}, Zl{k - 1}', 1);
            Zl{k} = unfold(tmp, 'left')' * unfold(Z.U{k}, 'left');
        end

        ZZ = cell(1, d); % final cores for normal (I \otimes X)Z(X^T)
        Zv = cell(1, d); % final cores for (I \otimes X)Z(V^T)
        vZ = cell(1, d); % final cores for (I \otimes V)Z(X^T)


        % contract to first core
        ZZ{1} = tensorprod_ttemps(Z.U{1}, Zr{2}, 3);
        % contract to inner cores
        for k = 2:(d - 1)
            res = tensorprod_ttemps(Z.U{k}, Zl{k - 1}, 1);
            ZZ{k} = tensorprod_ttemps(res, Zr{k + 1}, 3);
        end

        % contract to last core
        ZZ{d} = tensorprod_ttemps(Z.U{d}, Zl{d - 1}, 1);

        Zv{1} = tensorprod_ttemps(Z.U{1}, ZVr{2}, 3);

        for k = 2:(d - 1)
            res = tensorprod_ttemps(Z.U{k}, Zl{k - 1}, 1);
            Zv{k} = tensorprod_ttemps(res, ZVr{k + 1}, 3);
        end

        Zv{d} = tensorprod_ttemps(Z.U{d}, Zl{d - 1}, 1);


        vZ{1} = tensorprod_ttemps(Z.U{1}, Zr{2}, 3);

        for k = 2:(d - 1)
            res = tensorprod_ttemps(Z.U{k}, ZVl{k - 1}, 1);
            vZ{k} = tensorprod_ttemps(res, Zr{k + 1}, 3);
        end

        vZ{d} = tensorprod_ttemps(Z.U{d}, ZVl{d - 1}, 1);

        % Left unfold everything to apply additional operations
        for k = 1:d
            ZZ{k} = unfold(ZZ{k}, 'left');
            Zv{k} = unfold(Zv{k}, 'left');
            vZ{k} = unfold(vZ{k}, 'left');
        end
        % %%%%%%%%%%%%%%%%%%%%%%%%% TESTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        % 1st and last cases special, so they're computed outside the loop
        UL  = unfold(V.U{1}, 'left');
        dUL = unfold(dUR{1}, 'left');

        % right variational term without (I - UU^T) projection
        rightV = (Zv{1} - ZZ{1} * XtVg{2}) / R{1};

        Y.dU{1} = -dUL * UL' * ZZ{1} + rightV - UL * (UL' * rightV);

        for k = 2:(d - 1)
            UL  = unfold(V.U{k}, 'left');
            dUL = unfold(dUR{k}, 'left');

            % right variational term without (I - UU^T) projection
            rightV = (Zv{k} - ZZ{k} * XtVg{k + 1}) / R{k};

            Y.dU{k} = vZ{k} - UL * (UL' * vZ{k}) - dUL * UL' * ZZ{k} ...
                       + rightV - UL * (UL' * rightV);
        end

        Y.dU{d} = vZ{d};

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%                  CROSS TERMS                 %%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for k = 1:(d - 1)
            U = unfold(V.U{k}, 'left');
            ZZ{k} = ZZ{k} - U * (U' * ZZ{k});
            ZZ{k} = reshape(ZZ{k}, [r(k), n(k), r(k + 1)]);
        end

        ZZ{d} = reshape(ZZ{d}, [r(d), n(d), 1]);

        % Double loop to represent P_k DP_m, 1 <= k < d
        for k = 1:(d - 1)

            UL = unfold(V.U{k}, 'left');
            dUL = unfold(dUR{k}, 'left');

            for m = 1:(k - 1)
                VtZ = crossTermVariational(k, m, ZZ, dUR{m}, V);
                projUVtZ = VtZ - UL * (UL' * VtZ);

                Y.dU{k} = Y.dU{k} - projUVtZ;
            end

            for m = (k + 1):d
                ZtX = crossTermMatrixRight(k + 1, m, ZZ, V);
                Y.dU{k} = Y.dU{k} + dUL * ZtX;
            end

        end

        % Final terms for k = d
        for m = 1:(d - 1)
            VtZ = crossTermVariational(d, m, ZZ, dUR{m}, V);
            Y.dU{d} = Y.dU{d} - VtZ;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%FINAL RESHAPE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for k = 1:d
            Y.dU{k} = reshape(Y.dU{k}, [r(k), n(k), r(k + 1)]);
        end

    elseif ~exist('ind', 'var')% Z is a full array

        %{
        Note that you can split up the full expression for the Weingarten map in
        the following way:

        1) Diagonal terms(P_i DP_i terms, or non - cross terms)
        a) This can be split into(left variational) + (right variational),
        which correspond to two terms in product rule expansion
        2) Cross terms(P_i DP_j, we show mathematically equivalent expression
        in companion paper.)
        a) This likewise can be split into(i < j) + (i > j)
        %}

        % Cell that represents the LEFT VARIATIONAL SIDE
        Zv = cell(1, d);
        % Cell that represents the RIGHT VARIATIONAL SIDE
        vZ = cell(1, d);

        % Initialization step
        Zv{d} = Z(:);
        vZ{d} = Z(:);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%               DIAGONAL TERMS                 %%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%%%%%%%               LEFT VARIATIONAL               %%%%%%%%%%%%%%%%%%

        % First, calculate where the right side is the variational matrix
        % Note that the calculation here is split between VR^-1 and XX^tVR^-1
        % in (I - XX^t)VR^{-1} through xx and xx2 terms respectively

        % (d-1) term done separately to initialize the temp variable handoff
        zz = reshape(Z, [prod(n(1:(d - 1))), n(d)]);
        xx = transpose(unfold(V.dU{d}, 'right'));
        xxTmp = transpose(unfold(V.U{d}, 'right')); % temp var to help calc xx terms
        xx2 = transpose(unfold(V.V{d}, 'right'));

        % Z-matrix-k-variational, represents the xx calculation for Zv{k}
        Zmkv = zz * xx;
        vZ{d - 1} = zz * xx2;
        Zv{d - 1} = (Zmkv - vZ{d - 1} * XtVg{d}) / R{d - 1};

        % auxillery term for computing anything with variational matrix
        ZZtmp = zz * xxTmp;

        % Main calcualtion for right size of Zv
        for k = (d - 2):-1:1
            zz = reshape(Zmkv, [prod(n(1:k)), n(k + 1) * r(k + 2)]);
            xx = transpose(unfold(V.U{k + 1}, 'right'));
            Zmkv = zz * xx;

            zzTmp = reshape(ZZtmp, [prod(n(1:k)), n(k + 1) * r(k + 2)]);
            xxTmp = transpose(unfold(dUR{k + 1}, 'right'));
            Zmkv = Zmkv + zzTmp * xxTmp;
            ZZtmp = zzTmp * xx;

            % NOTE: vZ is only used here to store these values for later use
            % It stores the standard right side calc for the orthogonal projector
            zz2 = reshape(vZ{k + 1}, [prod(n(1:k)), n(k + 1) * r(k + 2)]);
            xx2 = transpose(unfold(V.V{k + 1}, 'right'));
            vZ{k} = zz2 * xx2;

            Zv{k} = (Zmkv - vZ{k} * XtVg{k + 1}) / R{k};
        end

        % Standard left side calculations for Zv
        for k = 2:d

            for i = 1:(k - 1)
                Z_i = reshape(Zv{k}, [r(i) * n(i), prod(n(i + 1:k)) * r(k + 1)]);
                X_i = unfold(V.U{i}, 'left');
                Zm = X_i' * Z_i;
                Zv{k} = Zm;
            end

            Zv{k} = reshape(Zv{k}, [r(k) * n(k), r(k + 1)]);
        end

        % Final U core projector mult for Zv
        for k = 1:(d - 1)
            U = unfold(V.U{k}, 'left');
            Zv{k} = Zv{k} - U * (U' * Zv{k});
        end

        %%%%%%%%%%%%%%%               RIGHT VARIATIONAL               %%%%%%%%%%%%%%%%%%

        % represents standard projection, also needed as intermediary for this section
        ZZ = vZ;
        % Start left side calculation for vZ
        for k = 2:d

            % Do first in loop separately to initialize intermediary variable
            Z_i = reshape(vZ{k}, [n(1), prod(n(2:k)) * r(k + 1)]);
            dU_i = unfold(dUR{1}, 'left');
            Zm = dU_i' * Z_i;
            vZ{k} = Zm;

            X_i = unfold(V.U{1}, 'left');
            ZZ{k} = X_i' * Z_i;

            for m = 2:(k - 1)
                Z_i = reshape(vZ{k}, [r(m) * n(m), prod(n(m + 1:k)) * r(k + 1)]);
                X_i = unfold(V.U{m}, 'left');
                Zm = X_i' * Z_i;

                Z_tmp = reshape(ZZ{k}, [r(m) * prod(n(m)), prod(n(m + 1:k)) * r(k + 1)]);
                dU_i = unfold(dUR{m}, 'left');

                vZ{k} = Zm + dU_i' * Z_tmp;

                ZZ{k} = X_i' * Z_tmp;
            end

            vZ{k} = reshape(vZ{k}, [r(k) * n(k), r(k + 1)]);
            ZZ{k} = reshape(ZZ{k}, [r(k) * n(k), r(k + 1)]);
        end

        % Final U core projector mult for vZ

        % 1st core separate since vZ{1} = 0 up until now. Purely for efficiency
        U  = unfold(V.U{1},  'left');
        dU = unfold(V.dU{1}, 'left');
        vZ{1} = -dU * U' * ZZ{1};

        for k = 2:(d - 1)
            U  = unfold(V.U{k},  'left');
            dU = unfold(V.dU{k}, 'left');
            vZ{k} = vZ{k} - U * (U' * vZ{k});

            vZ{k} = vZ{k} - dU * U' * ZZ{k};

        end

        for k = 1:d - 1
            Y.dU{k} = Zv{k} + vZ{k};
        end

        Y.dU{d} = vZ{d};


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%                  CROSS TERMS                 %%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for k = 1:(d - 1)
            % now only need ZZ as dU cores of normal projection of V
            U = unfold(V.U{k}, 'left');
            ZZ{k} = ZZ{k} - U * (U' * ZZ{k});
            ZZ{k} = reshape(ZZ{k}, [r(k), n(k), r(k + 1)]);
        end

        ZZ{d} = reshape(ZZ{d}, [r(d), n(d), 1]);

        % Double loop to represent P_k DP_m, 1 <= k < d
        for k = 1:(d - 1)

            UL  = unfold(V.U{k},  'left');
            dUL = unfold(V.dU{k}, 'left');

            for m = 1:(k - 1)
                VtZ = crossTermVariational(k, m, ZZ, dUR{m}, V);

                projUVtZ = VtZ - UL * (UL' * VtZ);

                Y.dU{k} = Y.dU{k} - projUVtZ; 
            end

            for m = (k + 1):d
                ZtX = crossTermMatrixRight(k + 1, m, ZZ, V);
                Y.dU{k} = Y.dU{k} + dUL * ZtX;
            end

        end

        % Final terms for k = d
        for m = 1:(d - 1)
            VtZ = crossTermVariational(d, m, ZZ, dUR{m}, V);

            Y.dU{d} = Y.dU{d} - VtZ;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FINAL RESHAPE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for k = 1:d
            Y.dU{k} = reshape(Y.dU{k}, [r(k), n(k), r(k + 1)]);
        end

    else
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SPARSE CASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % permuted U, V, and dU cells for the efficient computation in C
        CU = cell(1, d);
        CV = cell(1, d);
        CdUR = cell(1, d);

        for k = 1:d
            CU{k} = permute(V.U{k}, [1 3 2]);
            CV{k} = permute(V.V{k}, [1 3 2]);
            CdUR{k} = permute(dUR{k}, [1 3 2]);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%               DIAGONAL TERMS                 %%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        [ZZ, vZ, Zv] = weingarten_omega(n, r, CU, CV, CdUR, ind.', Z);

        for k = 1:d

            ZZ{k} = reshape(ZZ{k}, [r(k), r(k + 1), n(k)]);
            ZZ{k} = unfold(permute(ZZ{k}, [1 3 2]), 'left');

            vZ{k} = reshape(vZ{k}, [r(k), r(k + 1), n(k)]);
            vZ{k} = unfold(permute(vZ{k}, [1 3 2]), 'left');

            Zv{k} = reshape(Zv{k}, [r(k), r(k + 1), n(k)]);
            Zv{k} = unfold(permute(Zv{k}, [1 3 2]), 'left');
        end


            

        % 1st and last cases special, so they're computed outside the loop
        UL  = unfold(V.U{1}, 'left');
        dUL = unfold(dUR{1}, 'left');

        % right variational term without (I - UU^T) projection
        rightV = (Zv{1} - ZZ{1} * XtVg{2}) / R{1};
        Y.dU{1} = -dUL * UL' * ZZ{1} + rightV - UL * (UL' * rightV);

        for k = 2:(d - 1)
            UL  = unfold(V.U{k}, 'left');
            dUL = unfold(dUR{k}, 'left');

            % right variational term without (I - UU^T) projection
            rightV = (Zv{k} - ZZ{k} * XtVg{k + 1}) / R{k};
            Y.dU{k} = vZ{k} - UL * (UL' * vZ{k}) - dUL * UL' * ZZ{k} + rightV - UL * (UL' * rightV);
            % norm(vZ{k} - UL * (UL' * vZ{k}) + (Zv{k} - UL * (UL' * Zv{k})) / R{k}) / norm(V)
            % norm(V)
        end


        Y.dU{d} = vZ{d};


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%                  CROSS TERMS                 %%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for k = 1:(d - 1)
            U = unfold(V.U{k}, 'left');
            ZZ{k} = ZZ{k} - U * (U' * ZZ{k});
            ZZ{k} = reshape(ZZ{k}, [r(k), n(k), r(k + 1)]);
        end

        ZZ{d} = reshape(ZZ{d}, [r(d), n(d), 1]);

        % Double loop to represent P_k DP_m, 1 <= k < d
        for k = 1:(d - 1)

            UL  = unfold(V.U{k}, 'left');
            dUL = unfold(dUR{k}, 'left');

            for m = 1:(k - 1)
                % XtZ = crossTermMatrixLeft(k, m, ZZ, V);
                VtZ = crossTermVariational(k, m, ZZ, dUR{m}, V);
                projUVtZ = VtZ - UL * (UL' * VtZ);

                Y.dU{k} = Y.dU{k} - projUVtZ; %+ dUL * XtZ;

                % Y.dU{k} = Y.dU{k} - (projUk * kron(eye(n(k)), Vllk') - dUkL * Xlek') * Zkm;
            end

            for m = (k + 1):d
                ZtX = crossTermMatrixRight(k + 1, m, ZZ, V);
                Y.dU{k} = Y.dU{k} + dUL * ZtX;

                %         Y.dU{k} = Y.dU{k} + dUkL * Zkm' * Xg{k+1};
            end

        end

        % Final terms for k = d
        for m = 1:(d - 1)
            VtZ = crossTermVariational(d, m, ZZ, dUR{m}, V);

            Y.dU{d} = Y.dU{d} - VtZ;

            %     Y.dU{d} = Y.dU{d} - kron(eye(n(d)), Vlld') * Zkm;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%FINAL RESHAPE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for k = 1:d
            Y.dU{k} = reshape(Y.dU{k}, [r(k), n(k), r(k + 1)]);
        end

        Y = (normV) * Y;
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Efficient ZtX term for P_k DP_m, m > k
% Zp here is the projection of Z (just dU cores)
% V is only passed for access to U and V cores
function ZtX = crossTermMatrixRight(k, m, Zp, V)
    ZtX = conj(unfold(Zp{m}, 'right')) * unfold(V.V{m}, 'right').';

    for p = (m - 1):-1:k
        tmp = tensorprod_ttemps(V.U{p}, (ZtX)', 3);
        ZtX = conj(unfold(tmp, 'right')) * unfold(V.V{p}, 'right').';
    end

end

% Efficient XtZ term for P_k DP_m, m < k
% Zp here is the projection of Z
% V is only passed for access to U and V cores
function XtZ = crossTermMatrixLeft(k, m, Zp, V) %#ok<DEFNU>

    XtZ = unfold(V.U{m}, 'left')' * unfold(Zp{m}, 'left');

    for p = (m + 1):k
        XtZ = tensorprod_ttemps(V.U{p}, (XtZ)', 1);
        XtZ = unfold(XtZ, 'left')' * unfold(V.V{p}, 'left');
    end

end

% Efficient VtZ (more accurately kron(eye(n(k)), V') * Z) term for P_k DP_m, m < k
% Zp here is the projection of Z, dURm is normally parametrized dU{m}
% V is only passed for access to U and V cores
function VtZ = crossTermVariational(k, m, Zp, dURm, V)

    VtZ = unfold(dURm, 'left')' * unfold(Zp{m}, 'left');

    for p = (m + 1):(k - 1)
        VtZ = tensorprod_ttemps(V.U{p}, (VtZ)', 1);
        VtZ = unfold(VtZ, 'left')' * unfold(V.V{p}, 'left');
    end

    VtZ = tensorprod_ttemps(V.V{k}, VtZ, 1);
    VtZ = unfold(VtZ, 'left');

end
