classdef anisotropicdiffusion < TTeMPS_op_laplace
    % Class for anisotropic diffusion operator with tridiagonal diffusion matrix
    %
    %       [ 1 a 0     ...0 ]
    %       [ a 1 a 0    ..0 ]
    %   D = [ 0 a 1 a 0  ..0 ]
    %       [  ..   .. . .   ]
    %       [ 0 ... .. 0 a 1 ]
    %
    %

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    properties
        L
        D
        % precomputed spectral decomp of 1D Laplace:

    end

    methods

        function A = update_properties( A );

            A.rank = [1,  3*ones(1, length(A.U)-1), 1];  % the TT rank is always three for such Laplace-like tensors
            size_col_ = cellfun( @(y) size(y,1), A.U);
            A.size_col = size_col_ ./ (A.rank(1:end-1).*A.rank(2:end));
            A.size_row = cellfun( @(y) size(y,2), A.U);
            A.order = length( A.size_row );
        end
    end


    methods( Access = public )

        function A = anisotropicdiffusion( n, d, alpha )

            if ~exist('alpha', 'var')
                alpha = 0.25;
            end
            
            one = ones(n,1);
            q = linspace( -10, 10, n)';
            h = abs(q(2) - q(1));
            L = -spdiags( [one, -2*one, one], [-1 0 1], n, n) / (h^2);

            % superclass constructor
            A = A@TTeMPS_op_laplace( L, d );
            % precompute eigenvalue information and exponential for use in local
            A = initialize_precond( A );
            % preconditioner:
            A.L = L;

			[A.V_L, A.E_L] = eig(full(A.L));
            A.E_L = diag(A.E_L);

            A.D = spdiags( [-one,one], [-1,1], n, n ) / (2*h); 
            I = speye( n, n );

            e1 = sparse( 1, 1, 1, 3, 1 );
            e2 = sparse( 2, 1, 1, 3, 1 );
            e3 = sparse( 3, 1, 1, 3, 1 );

            l_mid = sparse( 3, 1, 1, 9, 1 );                % e_3
            b_mid = sparse( 6, 1, 1, 9, 1 );                % e_6
            m_mid = sparse( [1;9], [1;1], [1;1], 9, 1 );    % e_1 + e_9
            c_mid = sparse( 2, 1, 1, 9, 1 );                % e_2

            A.U = cell( 1, d );
            A.U{1} = kron( A.L, e1 ) + kron( 2*alpha*A.D, e2 ) + kron( I, e3);
            A_mid = kron( A.L, l_mid ) + kron( 2*alpha*A.D, b_mid ) + kron( I, m_mid) + kron( A.D, c_mid );
            for i=2:d-1
                A.U{i} = A_mid;
            end
            A.U{d} = kron( I, e1 ) + kron( A.D, e2 ) + kron( A.L, e3);

            A = update_properties( A );
           
        end

        function expB = constr_precond_inner( A, X, mu )

            n = size(A.L, 1);
            sz = [X.rank(mu), X.size(mu), X.rank(mu+1)]

            B1 = zeros( X.rank(mu) );
            % calculate B1 part:
            for i = 1:mu-1
                % apply L to the i'th core
                tmp = X;
                Xi = matricize( tmp.U{i}, 2 );
                Xi = A.L*Xi;
                tmp.U{i} = tensorize( Xi, 2, [X.rank(i), n, X.rank(i+1)] );
                B1 = B1 + innerprod( X, tmp, 'LR', mu-1);
            end

            B3 = zeros( X.rank(mu+1) );
            % calculate B3 part:
            for i = mu+1:A.order
                tmp = X;
                Xi = matricize( tmp.U{i}, 2 );
                Xi = A.L*Xi;
                tmp.U{i} = tensorize( Xi, 2, [X.rank(i), n, X.rank(i+1)] );
                B3 = B3 + innerprod( X, tmp, 'RL', mu+1);
            end
            
            [V1,e1] = eig(B1);
            e1 = diag(e1);
            [V3,e3] = eig(B3);
            e3 = diag(e3);

            lmin = min(e1) + min(A.E_L) + min(e3);
            lmax = max(e1) + max(A.E_L) + max(e3);

            R = lmax/lmin
            
            [omega, alpha] = load_coefficients( R );

            k = 3;
            omega = omega/lmin;
            alpha = alpha/lmin;

            expB = cell(3,k);
            
            for i = 1:k
                expB{1,i} = omega(i) * V1*diag( exp( -alpha(i)*e1 ))*V1';    % include omega in first part
                expB{2,i} = A.V_L*diag( exp( -alpha(i)*A.E_L ))*A.V_L';
                expB{3,i} = V3*diag( exp( -alpha(i)*e3 ))*V3';
            end
        end

        function expB = constr_precond_outer( A, X, mu1, mu2 )
            
            n = size(A.L, 1);

            B1 = zeros( X.rank(mu1) );
            % calculate B1 part:
            for i = 1:mu1-1
                % apply L to the i'th core
                tmp = X;
                Xi = matricize( tmp.U{i}, 2 );
                Xi = A.L*Xi;
                tmp.U{i} = tensorize( Xi, 2, [X.rank(i), n, X.rank(i+1)] );
                B1 = B1 + innerprod( X, tmp, 'LR', mu1-1);
            end

            B3 = zeros( X.rank(mu2+1) );
            % calculate B3 part:
            for i = mu2+1:A.order
                tmp = X;
                Xi = matricize( tmp.U{i}, 2 );
                Xi = A.L*Xi;
                tmp.U{i} = tensorize( Xi, 2, [X.rank(i), n, X.rank(i+1)] );
                B3 = B3 + innerprod( X, tmp, 'RL', mu2+1);
            end
            
            [V1,e1] = eig(B1);
            e1 = diag(e1);
            [V3,e3] = eig(B3);
            e3 = diag(e3);

            lmin = min(e1) + 2*min(A.E_L) + min(e3);
            lmax = max(e1) + 2*max(A.E_L) + max(e3);

            R = lmax/lmin
            
            [omega, alpha] = load_coefficients( R );

            k = 3;
            omega = omega/lmin;
            alpha = alpha/lmin;

            expB = cell(4,k);
            
            for i = 1:k
                expB{1,i} = omega(i) * V1*diag( exp( -alpha(i)*e1 ))*V1';    % include omega in first part
                expB{2,i} = A.V_L*diag( exp( -alpha(i)*A.E_L ))*A.V_L';
                expB{3,i} = A.V_L*diag( exp( -alpha(i)*A.E_L ))*A.V_L';
                expB{4,i} = V3*diag( exp( -alpha(i)*e3 ))*V3';
            end
        end

        function P = constr_precond( A, k )

            d = A.order;

            lmin = d*min(A.E_L);
            lmax = d*max(A.E_L);

            R = lmax/lmin

            %  http://www.mis.mpg.de/scicomp/EXP_SUM/1_x/1_xk07_2E2
            %  0.0133615547183825570028305575534521842940   {omega[1]}
            %  0.0429728469424360175410925952177443321034   {omega[2]}
            %  0.1143029399081515586560726591147663100401   {omega[3]}
            %  0.2838881266934189482611071431161775535656   {omega[4]}
            %  0.6622322841999484042811198458711174907876   {omega[5]}
            %  1.4847175320092703810050463464342840325116   {omega[6]}
            %  3.4859753729916252771962870138366952232900   {omega[7]}
            %  0.0050213411684266507485648978019454613531   {alpha[1]}
            %  0.0312546410994290844202411500801774835168   {alpha[2]}
            %  0.1045970270084145620410366606112262388706   {alpha[3]}
            %  0.2920522758702768403556507270657505159761   {alpha[4]}
            %  0.7407504784499061527671195936939341208927   {alpha[5]}
            %  1.7609744335543204401530945069076494746696   {alpha[6]}
            %  4.0759036969145123916954953635638503328664   {alpha[7]}
            
            if k == 3
                [omega, alpha] = load_coefficients( R );

            elseif k == 7
                omega = [0.0133615547183825570028305575534521842940 0.0429728469424360175410925952177443321034 0.1143029399081515586560726591147663100401,...
                         0.2838881266934189482611071431161775535656 0.6622322841999484042811198458711174907876 1.4847175320092703810050463464342840325116,...
                         3.4859753729916252771962870138366952232900];
                alpha = [0.0050213411684266507485648978019454613531 0.0312546410994290844202411500801774835168 0.1045970270084145620410366606112262388706,...
                         0.2920522758702768403556507270657505159761 0.7407504784499061527671195936939341208927 1.7609744335543204401530945069076494746696,...
                         4.0759036969145123916954953635638503328664];
            else
                error('Unknown rank specified. Choose either k=3 or k=7');
            end

            omega = omega/lmin;
            alpha = alpha/lmin;

            % careful: all cores assumed to be of same size
            E = reshape( expm( -alpha(1) * A.L), [1, A.size_row(2), A.size_col(2), 1]);
            P = omega(1)*TTeMPS_op( repmat({E},1,d) );
            for i = 2:k
                E = reshape( expm( -alpha(i) * A.L), [1, A.size_row(2), A.size_col(2), 1]);
                P = P + omega(i)*TTeMPS_op( repmat({E},1,d) );
            end

        end

    end
        
end
