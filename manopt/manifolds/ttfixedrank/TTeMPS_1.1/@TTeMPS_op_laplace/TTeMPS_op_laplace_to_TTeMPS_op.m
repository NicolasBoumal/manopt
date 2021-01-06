function res = TTeMPS_op_laplace_to_TTeMPS_op( A )
    %TTeMPS_op_laplace_to_TTeMPS Convert to TT_op_laplace to TTeMPS_op format.
    %
    %   See also TTeMPS_to_TT, TTeMPS_op_to_TT, TTeMPS_op_laplace_to_TTeMPS_op.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt
    
    C = cell(1, A.order);
    for i = 1:A.order
        % make a 4D tensor out of it again
        tmp = reshape( full(A.U{i}), [A.rank(i), A.rank(i+1), A.size_col(i), A.size_row(i)] );
        % inverse permute the indices
        C{i} = ipermute( tmp, [1 4 2 3] );
    end
    res = TTeMPS_op(C);
end
