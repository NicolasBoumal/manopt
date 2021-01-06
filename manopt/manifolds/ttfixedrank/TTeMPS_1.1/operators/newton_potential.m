classdef newton_potential < TTeMPS_op_laplace
    % Class for the Newton potential derived from the TTeMPS_op_laplace class.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    properties
        potential
        laplacerank
    end

    methods( Access = public )

        function A = newton_potential( n, d )
            
            one = ones(n,1);
            q = linspace( -1, 1, n)';
            h = abs(q(2) - q(1));
            L = -spdiags( [one, -2*one, one], [-1 0 1], n, n) / (h^2);            

            % superclass constructor
            A = A@TTeMPS_op_laplace( L, d );
            [A.V_L, A.E_L] = eig(full(A.L0));
            A.E_L = diag( A.E_L );

            % create potential

            a = d*min(q.^2);
            b = d*max(q.^2);

            k = 10;
            %  http://www.mis.mpg.de/scicomp/EXP_SUM/1_sqrtx/1_sqrtxk10_2E4
            %  0.0122353280385689973405283041685276401722   {omega[1]}
            %  0.0170048221567127200615394955196535420328   {omega[2]}
            %  0.0292580238127349109583679896348651361393   {omega[3]}
            %  0.0513674405142589591607559396796434114663   {omega[4]}
            %  0.0876522649954556273282240720645663856203   {omega[5]}
            %  0.1452687747918532464746523003018552344656   {omega[6]}
            %  0.2347776097665007749814821205736059539504   {omega[7]}
            %  0.3716699993620031342879666408363092955369   {omega[8]}
            %  0.5822367252037624550361008535226403637353   {omega[9]}
            %  0.9561416473224761119619093119315067497155   {omega[10]}
            %  0.0000278884968834587275257159520245177180   {alpha[1]}
            %  0.0003161367116046333453815662397571456532   {alpha[2]}
            %  0.0014174384495922332182950292714732065669   {alpha[3]}
            %  0.0052598987752533055729865546674972609509   {alpha[4]}
            %  0.0176506185630681001889793731510214236380   {alpha[5]}
            %  0.0548279527792261968593219480933020903990   {alpha[6]}
            %  0.1597705718538013465668734189306654513985   {alpha[7]}
            %  0.4411611832132119288894626929486975086547   {alpha[8]}
            %  1.1661865464705391523319785718193486445671   {alpha[9]}
            %  3.0303805868035630160465393467816852535179   {alpha[10]}

            omega = [0.0122353280385689973405283041685276401722; 0.0170048221567127200615394955196535420328;
                    0.0292580238127349109583679896348651361393; 0.0513674405142589591607559396796434114663;
                    0.0876522649954556273282240720645663856203; 0.1452687747918532464746523003018552344656;
                    0.2347776097665007749814821205736059539504; 0.3716699993620031342879666408363092955369;
                    0.5822367252037624550361008535226403637353; 0.9561416473224761119619093119315067497155];
            alpha = [0.0000278884968834587275257159520245177180; 0.0003161367116046333453815662397571456532;
                    0.0014174384495922332182950292714732065669; 0.0052598987752533055729865546674972609509;
                    0.0176506185630681001889793731510214236380; 0.0548279527792261968593219480933020903990;
                    0.1597705718538013465668734189306654513985; 0.4411611832132119288894626929486975086547;
                    1.1661865464705391523319785718193486445671; 3.0303805868035630160465393467816852535179];

            alpha = alpha/a;
            omega = omega/sqrt(a);

            C = cell(1, d);

            C = exp(-alpha(1)*q.^2);
            C = reshape( C, [1 n 1] );
            A.potential = omega(1)*TTeMPS( repmat( {C}, [1, d]));
            for i = 2:k
                C = exp(-alpha(i)*q.^2);
                C = reshape( C, [1 n 1] );
                A.potential = A.potential + omega(i)*TTeMPS( repmat( {C}, [1, d]));
            end

            A.laplacerank = A.rank;
            A.rank(2:end-1) = A.rank(2:end-1) + A.potential.rank(2:end-1);
            
        end


        
        function y = apply(A, x, idx)
            
            if ~exist( 'idx', 'var' )
                % apply laplace part
                A.rank = A.laplacerank;
                y = apply@TTeMPS_op_laplace( A, x );
                A.rank(2:end-1) = A.rank(2:end-1) + A.potential.rank(2:end-1);
                
                % apply henon-heiles part

                y = y + hadamard( A.potential, x );
                %y = hadamard( A.potential, x );

            else
                % apply laplace part
                A.rank = A.laplacerank;
                l = apply@TTeMPS_op_laplace( A, x, idx );
                A.rank(2:end-1) = A.rank(2:end-1) + A.potential.rank(2:end-1);
                
                % apply henon-heiles part

                h = hadamard( A.potential, x, idx);

                if idx == 1
                    y = zeros( 1, size(l,2), size(l,3)+size(h,3), size(h,4));
                    y( 1, :, 1:size(l,3), : ) = l;
                    y( 1, :, size(l,3)+1:end, : ) = h;
                elseif idx == A.order
                    y = zeros( size(l,1)+size(h,1), size(l,2), 1, size(h,4) );
                    y( 1:size(l,1), :, 1, :) = l;
                    y( size(l,1)+1:end, :, 1, :) = h;
                else
                    y = zeros( size(l,1)+size(h,1), size(l,2), size(l,3)+size(h,3), size(h,4) );
                    y( 1:size(l,1), :, 1:size(l,3), : ) = l;
                    y( size(l,1)+1:end, :, size(l,3)+1:end, : ) = h;
                end

            end
        end
        
       
        
        

        function P = constr_precond( A, k )

            d = A.order;
            ev = eig(full(A.L0));

            lmin = d*min(ev);
            lmax = d*max(ev);

            R = lmax/lmin;

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

            E = reshape( expm( -alpha(1) * A.L0), [1, A.size_row(d), A.size_col(d), 1]);
            P = omega(1)*TTeMPS_op( repmat({E},1,d) );
            for i = 2:k
                E = reshape( expm( -alpha(i) * A.L0), [1, A.size_row(d), A.size_col(d), 1]);
                P = P + omega(i)*TTeMPS_op( repmat({E},1,d) );
            end

        end

        function expB = constr_precond_inner( A, X, mu )
            
            %V = reshape( V, [1, 1, size(L, 1), tmp.rank(i), tmp.rank(i+1)] );
            %V = permute( V, [1, 4, 3, 2, 5]);
            %tmp.U{i} = reshape( V, [tmp.rank(i), size(L, 1), tmp.rank{i+1}]);

            n = size(A.L0, 1);
            sz = [X.rank(mu), X.size(mu), X.rank(mu+1)];

            B1 = zeros( X.rank(mu) );
            % calculate B1 part:
            for i = 1:mu-1
                % apply L to the i'th core
                tmp = X;
                Xi = matricize( tmp.U{i}, 2 );
                Xi = A.L0*Xi;
                tmp.U{i} = tensorize( Xi, 2, [X.rank(i), n, X.rank(i+1)] );
                B1 = B1 + innerprod( X, tmp, 'LR', mu-1);
            end

            % calculate B2 part:
            B2 = A.L0;

            B3 = zeros( X.rank(mu+1) );
            % calculate B3 part:
            for i = mu+1:A.order
                tmp = X;
                Xi = matricize( tmp.U{i}, 2 );
                Xi = A.L0*Xi;
                tmp.U{i} = tensorize( Xi, 2, [X.rank(i), n, X.rank(i+1)] );
                B3 = B3 + innerprod( X, tmp, 'RL', mu+1);
            end
            
            [V1,e1] = eig(B1);
            e1 = diag(e1);
            [V3,e3] = eig(B3);
            e3 = diag(e3);

            lmin = min(e1) + min(A.E_L) + min(e3);
            lmax = max(e1) + max(A.E_L) + max(e3);

            R = lmax/lmin;
            
            [omega, alpha] = load_coefficients( R );

            k = length(omega);
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
            
            %V = reshape( V, [1, 1, size(L, 1), tmp.rank(i), tmp.rank(i+1)] );
            %V = permute( V, [1, 4, 3, 2, 5]);
            %tmp.U{i} = reshape( V, [tmp.rank(i), size(L, 1), tmp.rank{i+1}]);

            n = size(A.L0, 1);

            B1 = zeros( X.rank(mu1) );
            % calculate B1 part:
            for i = 1:mu1-1
                % apply L to the i'th core
                tmp = X;
                Xi = matricize( tmp.U{i}, 2 );
                Xi = A.L0*Xi;
                tmp.U{i} = tensorize( Xi, 2, [X.rank(i), n, X.rank(i+1)] );
                B1 = B1 + innerprod( X, tmp, 'LR', mu1-1);
            end

            % calculate B2 part:
            %D = spdiags( ones(n,1), 0, n, n);
            %B2 = kron(A.L, D) + kron(D, A.L);

            B3 = zeros( X.rank(mu2+1) );
            % calculate B3 part:
            for i = mu2+1:A.order
                tmp = X;
                Xi = matricize( tmp.U{i}, 2 );
                Xi = A.L0*Xi;
                tmp.U{i} = tensorize( Xi, 2, [X.rank(i), n, X.rank(i+1)] );
                B3 = B3 + innerprod( X, tmp, 'RL', mu2+1);
            end
            
            [V1,e1] = eig(B1);
            e1 = diag(e1);
            [V3,e3] = eig(B3);
            e3 = diag(e3);

            lmin = min(e1) + 2*min(A.E_L) + min(e3);
            lmax = max(e1) + 2*max(A.E_L) + max(e3);

            R = lmax/lmin;
            
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

    end
        
end
