classdef TTeMPS_op_laplace
% TTeMPS_op_laplace
%
%   A MATLAB class for representing and manipulating 
%   Laplace-like operators in the TT/MPS operator format,
%
%   Laplace-like operators are of the form
%         L \otimes I \otimes I \otimes I 
%       + I \otimes L \otimes I \otimes I 
%       + ... 
%       + I \otimes I \otimes I \otimes L
% 
%   with e.g. L being the discrete 1D-Laplacian matrix. 

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

properties( SetAccess = public, GetAccess = public )

    L0
    U           % core tensors as 4D doubles
    rank
    order
    size_row
    size_col
    V_L
    E_L

end

% Set methods for Cores
methods
    
    function A = set.U( A, U_);
        
        A.U = U_;
        A = update_properties( A );

    end


    function A = update_properties( A );

        A.rank = [1,  2*ones(1, length(A.U)-1), 1];  % the TT rank is always two for such Laplace-like tensors
        size_col_ = cellfun( @(y) size(y,1), A.U);
        A.size_col = size_col_ ./ (A.rank(1:end-1).*A.rank(2:end));
        A.size_row = cellfun( @(y) size(y,2), A.U);
        A.order = length( A.size_row );

    end

end

methods( Access = public )

    function A = TTeMPS_op_laplace(varargin)
    %LAPLACE Construct a tensor in TT/MPS format and return TTeMPS_op_laplace object.
    %
    %   A = TTEMPS_OP_LAPLACE( L, D ) creates the D-dimensional Laplace-like
    %       TT/MPS operator using the supplied matrix L.
    %
    %
        if nargin == 1
            % debug constructor
            A.U = varargin{1};
            A = update_properties( A );
            A.L0 = [];
            A.V_L = [];
            A.E_L = [];
        % Default constructor
        elseif (nargin == 2)

            % only one matrix passed
            L = varargin{1};
            A.L0 = L;
            d = varargin{2};

            A.V_L = [];
            A.E_L = [];
            
            [m,n] = size( L );
            E = speye( m, n );
            a_1 = sparse( 1, 1, 1, 2, 1 );
            a_mid = sparse( 2, 1, 1, 4, 1 );
            a_end = sparse( 2, 1, 1, 2, 1 );
            b_1 = sparse( 2, 1, 1, 2, 1 );
            b_mid = sparse( [1;4], [1;1], [1;1], 4, 1 );
            b_end = sparse( 1, 1, 1, 2, 1 );

            A.U = cell( 1, d );
            A.U{1} = kron( L, a_1 ) + kron( E, b_1 );
            A_mid = kron( L, a_mid ) + kron( E, b_mid );
            for i=2:d-1
                A.U{i} = A_mid;
            end
            A.U{d} = kron( L, a_end ) + kron( E, b_end );

            A = update_properties( A );

        else
            error('Invalid number of arguments.')
        end
    end
    
    
    % Other public functions
    y = apply( A, x, idx );
    A = mtimes( B, A );
    res = contract( A, x, y, idx );
    
    disp( A, name );
    display( A );

    B = TTeMPS_op_laplace_to_TTeMPS_op( A );
    B = TTeMPS_op_laplace_to_TT_matrix( A );
    expB = constr_precond_inner( A, X, mu );
    

    function A = initialize_precond( A )
        [A.V_L, A.E_L] = eig(full(A.L0));
        A.E_L = diag( A.E_L );
    end
end



end
