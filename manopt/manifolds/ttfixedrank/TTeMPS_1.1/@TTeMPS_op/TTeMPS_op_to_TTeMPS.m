function B = TTeMPS_op_to_TTeMPS( A )
    %TTeMPS_op_to_TTeMPS Convert to TT Toolbox matrix format.
    %   TT = TTeMPS_op_to_TTeMPS( A ) takes the TTeMPS operator A and converts it into
    %   a TTeMPS object by reshaping.
    %
    %   See also TTeMPS_op_to_TT_matrix

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    U = cellfun(@(y) reshape(y, [size(y,1), size(y,2)*size(y,3), size(y,4)]), ...
                        A.U, 'UniformOutput', false);

    B = TTeMPS( U );
end
