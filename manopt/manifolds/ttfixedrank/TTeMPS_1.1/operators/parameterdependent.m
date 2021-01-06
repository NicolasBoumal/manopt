classdef parameterdependent
    % Operator from a parameter dependent heat equation,
    % the so-called cookie problem.
    % Supported number of cookies: 4 or 9.
    %
    % See the following paper, Section 5.3, for a description:
    %
    % D. Kressner, M. Steinlechner, and B. Vandereycken. 
    % Preconditioned low-rank Riemannian optimization for 
    % linear systems with tensor product structure. 
    % Technical report, July 2015. Revised February 2016. 
    % To appear in SIAM J. Sci. Comput. 
    % 

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    
    properties( SetAccess = protected, GetAccess = public )

        A           % cell storing the galerkin matrices A0,...Ad
        param
        rank
        order
        Lapl
        size_row
        size_col
    end

    methods( Access = public )

        function A = parameterdependent( n, numcookie )
            
            % only one matrix passed
            A.param = transpose(linspace(1,10,n));
            
            if numcookie == 4
                l = load('four_cookies.mat');
                A.A{1} = l.A0;
                A.A{2} = l.A1;
                A.A{3} = l.A2;
                A.A{4} = l.A3;
                A.A{5} = l.A4;
            
                A.rank = 5;
                A.order = 5;
                A.size_row = [1169,n,n,n,n];
                A.size_col = [1169,n,n,n,n];
                
                A.Lapl = parameter_to_TTeMPS_op_laplace( A );
                
            elseif numcookie == 9
                l = load('nine_cookies.mat');
                A.A{1} = l.A0;
                A.A{2} = l.A1;
                A.A{3} = l.A2;
                A.A{4} = l.A3;
                A.A{5} = l.A4;
                A.A{6} = l.A5;
                A.A{7} = l.A6;
                A.A{8} = l.A7;
                A.A{9} = l.A8;
                A.A{10} = l.A9;
            
                A.rank = 10;
                A.order = 10;
                A.size_row = [2796,n,n,n,n];
                A.size_col = [2796,n,n,n,n];
                
                A.Lapl = parameter_to_TTeMPS_op_laplace( A );
            else
                error('Only 4 or 9 cookies supported atm!')
            end
    
        end
        
        % Apply member function copied over from TTeMPS_op_laplace
        function y = apply( A, x, idx )
            %APPLY Application of TT/MPS parameter-dependent operator to a TT/MPS tensor
            %   Y = APPLY(A, X) applies the TT/MPS Laplace operator A to the TT/MPS tensor X.
            %
            %   Y = APPLY(A, X, idx) is the application of A but only in mode idx.
            %       note that in this case, X is assumed to be a standard matlab array and
            %       not a TTeMPS tensor. 
            %
            %   In both cases, X can come from a block-TT format, that is, with a four-dimensional core instead.
            %
            if ~exist( 'idx', 'var' )
                y = apply(A.Lapl, x );
            else
                y = apply(A.Lapl, x, idx);
            end
        end
        
        function B = parameter_to_TTeMPS_op_laplace( A )
            if A.order == 5
                a_0 = sparse( 1, 1, 1, 5, 1 );
                a_1 = sparse( 2, 1, 1, 5, 1 );
                a_2 = sparse( 3, 1, 1, 5, 1 );
                a_3 = sparse( 4, 1, 1, 5, 1 );
                a_4 = sparse( 5, 1, 1, 5, 1 );
                
                U = cell( 1, 5 );
                U{1} = kron( A.A{1}, a_0 ) + kron( A.A{2}, a_1 ) ...
                         + kron( A.A{3}, a_2 ) + kron( A.A{4}, a_3 ) ...
                         + kron( A.A{5}, a_4 ); 
                   
                n = length(A.param)      
                D = spdiags(A.param,0,n,n);
                E = speye(n);
                
                b_0 = sparse( 1 , 1, 1, 25, 1 );
                b_1 = sparse( 7 , 1, 1, 25, 1 );
                b_2 = sparse( 13, 1, 1, 25, 1 );
                b_3 = sparse( 19, 1, 1, 25, 1 );
                b_4 = sparse( 25, 1, 1, 25, 1 );
                
                U{2} = kron( E, b_0 ) + kron( D, b_1 ) ...
                       + kron( E, b_2 ) + kron( E, b_3 ) ...
                       + kron( E, b_4 );
                U{3} = kron( E, b_0 ) + kron( E, b_1 ) ...
                       + kron( D, b_2 ) + kron( E, b_3 ) ...
                       + kron( E, b_4 );
                U{4} = kron( E, b_0 ) + kron( E, b_1 ) ...
                       + kron( E, b_2 ) + kron( D, b_3 ) ...
                       + kron( E, b_4 );
                U{5} = kron( E, a_0 ) + kron( E, a_1 ) ...
                       + kron( E, a_2 ) + kron( E, a_3 ) ...
                       + kron( D, a_4 );      
                       
                B = TTeMPS_op_laplace( U );
                B.rank = [1 5 5 5 5 1];
                B.size_col = B.size_row;
            elseif A.order == 10
                a_0 = sparse( 1,  1, 1, 10, 1 );
                a_1 = sparse( 2,  1, 1, 10, 1 );
                a_2 = sparse( 3,  1, 1, 10, 1 );
                a_3 = sparse( 4,  1, 1, 10, 1 );
                a_4 = sparse( 5,  1, 1, 10, 1 );
                a_5 = sparse( 6,  1, 1, 10, 1 );
                a_6 = sparse( 7,  1, 1, 10, 1 );
                a_7 = sparse( 8,  1, 1, 10, 1 );
                a_8 = sparse( 9,  1, 1, 10, 1 );
                a_9 = sparse( 10, 1, 1, 10, 1 );
                
                U = cell( 1, 10 );
                U{1} = kron( A.A{1}, a_0 ) + kron( A.A{2}, a_1 ) ...
                         + kron( A.A{3}, a_2 ) + kron( A.A{4}, a_3 ) ...
                         + kron( A.A{5}, a_4 ) + kron( A.A{6}, a_5 ) ...
                         + kron( A.A{7}, a_6 ) + kron( A.A{8}, a_7 ) ...
                         + kron( A.A{9}, a_8 ) + kron( A.A{10}, a_9 );
                   
                n = length(A.param)      
                D = spdiags(A.param,0,n,n);
                E = speye(n);
                
                b_0 = sparse(   1, 1, 1, 100, 1 );
                b_1 = sparse(  12, 1, 1, 100, 1 );
                b_2 = sparse(  23, 1, 1, 100, 1 );
                b_3 = sparse(  34, 1, 1, 100, 1 );
                b_4 = sparse(  45, 1, 1, 100, 1 );
                b_5 = sparse(  56, 1, 1, 100, 1 );
                b_6 = sparse(  67, 1, 1, 100, 1 );
                b_7 = sparse(  78, 1, 1, 100, 1 );
                b_8 = sparse(  89, 1, 1, 100, 1 );
                b_9 = sparse( 100, 1, 1, 100, 1 );

                U{2} = kron( E, b_0 ) + kron( D, b_1 ) ...
                         + kron( E, b_2 ) + kron( E, b_3 ) ...
                         + kron( E, b_4 ) + kron( E, b_5 ) ...
                         + kron( E, b_6 ) + kron( E, b_7 ) ...
                         + kron( E, b_8 ) + kron( E, b_9 );
                U{3} = kron( E, b_0 ) + kron( E, b_1 ) ...
                         + kron( D, b_2 ) + kron( E, b_3 ) ...
                         + kron( E, b_4 ) + kron( E, b_5 ) ...
                         + kron( E, b_6 ) + kron( E, b_7 ) ...
                         + kron( E, b_8 ) + kron( E, b_9 );
                U{4} = kron( E, b_0 ) + kron( E, b_1 ) ...
                         + kron( E, b_2 ) + kron( D, b_3 ) ...
                         + kron( E, b_4 ) + kron( E, b_5 ) ...
                         + kron( E, b_6 ) + kron( E, b_7 ) ...
                         + kron( E, b_8 ) + kron( E, b_9 );
                U{5} = kron( E, b_0 ) + kron( E, b_1 ) ...
                         + kron( E, b_2 ) + kron( E, b_3 ) ...
                         + kron( D, b_4 ) + kron( E, b_5 ) ...
                         + kron( E, b_6 ) + kron( E, b_7 ) ...
                         + kron( E, b_8 ) + kron( E, b_9 );
                U{6} = kron( E, b_0 ) + kron( E, b_1 ) ...
                         + kron( E, b_2 ) + kron( E, b_3 ) ...
                         + kron( E, b_4 ) + kron( D, b_5 ) ...
                         + kron( E, b_6 ) + kron( E, b_7 ) ...
                         + kron( E, b_8 ) + kron( E, b_9 );
                U{7} = kron( E, b_0 ) + kron( E, b_1 ) ...
                         + kron( E, b_2 ) + kron( E, b_3 ) ...
                         + kron( E, b_4 ) + kron( E, b_5 ) ...
                         + kron( D, b_6 ) + kron( E, b_7 ) ...
                         + kron( E, b_8 ) + kron( E, b_9 );
                U{8} = kron( E, b_0 ) + kron( E, b_1 ) ...
                         + kron( E, b_2 ) + kron( E, b_3 ) ...
                         + kron( E, b_4 ) + kron( E, b_5 ) ...
                         + kron( E, b_6 ) + kron( D, b_7 ) ...
                         + kron( E, b_8 ) + kron( E, b_9 );
                U{9} = kron( E, b_0 ) + kron( E, b_1 ) ...
                         + kron( E, b_2 ) + kron( E, b_3 ) ...
                         + kron( E, b_4 ) + kron( E, b_5 ) ...
                         + kron( E, b_6 ) + kron( E, b_7 ) ...
                         + kron( D, b_8 ) + kron( E, b_9 );
                U{10} = kron( E, a_0 ) + kron( E, a_1 ) ...
                         + kron( E, a_2 ) + kron( E, a_3 ) ...
                         + kron( E, a_4 ) + kron( E, a_5 ) ...
                         + kron( E, a_6 ) + kron( E, a_7 ) ...
                         + kron( E, a_8 ) + kron( D, a_9 );
                B = TTeMPS_op_laplace( U );
                B.rank = [1 10*ones(1,9) 1];
                B.size_col = B.size_row;
            else
                error('Wrong number of cookies!');
            end

        end
    end         
        
end
