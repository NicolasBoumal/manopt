function Afull = full( A )
    %FULL Convert TTeMPS_op operator to full array
    %   X = FULL(A) converts the TTeMPS_op operator A to an (A.order)-dimensional full array.
    %
    %	Use with care! Result can easily exceed available memory.
    %

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    d = A.order;
    X = full( TTeMPS_op_to_TTeMPS(A) );
    permutevec = [1:d; d+1:2*d];
    permutevec = permutevec(:)';

    sizes = [A.size_col, A.size_row];
    sizes = sizes( permutevec );
    X = reshape( X, sizes(permutevec) );
    X = ipermute( X, permutevec );
    Afull = reshape( X, prod(A.size_col), prod(A.size_row) );
end
