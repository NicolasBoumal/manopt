function B = TTeMPS_op_laplace_to_TT_matrix( A )
    %TTeMPS_to_TT Convert to TT Toolbox matrix format.
    %   TT = TT_to_TTeMPS( A ) takes the TTeMPS Laplace operator A and converts it into
    %   a tt_matrix object using the TT Toolbox 2.x from Oseledets et al.
    %   This toolbox needs to be installed, of course.
    %
    %   See also TTeMPS_to_TT, TTeMPS_op_to_TT, TTeMPS_op_laplace_to_TTeMPS_op.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    B = TTeMPS_op_to_TT_matrix( TTeMPS_op_laplace_to_TTeMPS_op( A ));

end
