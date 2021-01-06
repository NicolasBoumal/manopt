function B = TTeMPS_op_to_TT_matrix( A )
    %TTeMPS_op_to_TT Convert to TT Toolbox matrix format.
    %   TT = TT_op_to_TTeMPS( A ) takes the TTeMPS operator A and converts it into
    %   a tt_matrix object using the TT Toolbox 2.x from Oseledets et al.
    %   This toolbox needs to be installed, of course.
    %
    %   See also TTeMPS_to_TT.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    TT = tt_tensor;
    TT.d = A.order;
    TT.n = [A.size_col.*A.size_row]';
    TT.r = A.rank';
    TT.core = cell2mat( cellfun(@(y) y(:).', A.U, 'UniformOutput', false) ).';
    lengths = TT.r(1:end-1) .* TT.n .* TT.r(2:end);
    TT.ps = cumsum( [1; lengths] );

    B = tt_matrix( TT, A.size_col', A.size_row' );

end
