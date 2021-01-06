classdef TTeMPS_op
% TTeMPS_op
%
%   A MATLAB class for representing and manipulating tensor operators
%   in the TT/MPS operator format. 
% 

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt


properties( SetAccess = private, GetAccess = public )

    U           % core tensors as 4D doubles
    rank
    order
    size_row
    size_col

end

% Set methods for Cores
methods
    
    function A = set.U( A, U_);
        
        A.U = U_;
        A = update_properties( A );

    end


    function A = update_properties( A );

        rank_ = cellfun( @(x) size(x,1), A.U);
        A.rank = [rank_, size(A.U{end},4)];
        A.size_col = cellfun( @(y) size(y,2), A.U);
        A.size_row = cellfun( @(y) size(y,3), A.U);
        A.order = length( A.size_row );

    end

end



methods( Access = public )

    function A = TTeMPS_op(varargin)
    %TTEMPS Construct a tensor in TT/MPS format and return TTeMPS object.
    %
    %   A = TTEMPS() creates a zero TT/MPS tensor 
    %
    %   A = TTEMPS(CORES) creates a TT/MPS tensor with core tensors C taken 
    %   from the cell array CORES
    %
    %   A = TTEMPS(CORES, ORTH) creates a TT/MPS tensor with core tensors C 
    %   taken from the cell array CORES. ORTH specifies the position of 
    %   orthogonalization (default = 0, no orthogonalization).
    %

        % Default constructor
        if (nargin == 0)
          
            A = TTeMPS_op( {0 0 0} );
            return;
          
        elseif (nargin == 1)

            % CAREFUL, add sanity check here
            A.U = varargin{1};

            A = update_properties( A );

        else
            error('Invalid number of arguments.')
        end
    end

    % Other public functions
    disp( A, name );
    display( A );

    res = contract( A, x, y, idx );
    B = TTeMPS_op_to_TT_matrix( A );
    B = TTeMPS_op_to_TTeMPS( A );
    y = apply( A, x, idx );
    A = round(A, tol );
    
    A = plus( A, B );
    A = mtimes( B, A );

    Afull = full( A );
    
end

end
