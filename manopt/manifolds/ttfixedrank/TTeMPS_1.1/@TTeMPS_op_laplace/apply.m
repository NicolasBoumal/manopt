function y = apply( A, x, idx )
    %APPLY Application of TT/MPS Laplace-like operator to a TT/MPS tensor
    %   Y = APPLY(A, X) applies the TT/MPS Laplace operator A to the TT/MPS tensor X.
    %
    %   Y = APPLY(A, X, idx) is the application of A but only in mode idx.
    %       note that in this case, X is assumed to be a standard matlab array and
    %       not a TTeMPS tensor. 
    %
    %   In both cases, X can come from a block-TT format, that is, with a four-dimensional core instead.
    %
    %   See also CONTRACT

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt


    % first case: all cores.
    if ~exist( 'idx', 'var' )
        V = cell(1, A.order);

        for i = 1:A.order
            %check for possible block format
            p = size(x.U{i},4);
            if p ~= 1
                Xi = permute( x.U{i}, [2 1 3 4]);
                Xi = reshape( Xi, [A.size_row(i), x.rank(i)*x.rank(i+1)*p] );
                V{i} = A.U{i}*Xi;
                V{i} = reshape( V{i}, [A.rank(i), A.rank(i+1), A.size_col(i), x.rank(i), x.rank(i+1), p] );
                V{i} = permute( V{i}, [1, 4, 3, 2, 5, 6]);
                V{i} = reshape( V{i}, [A.rank(i)*x.rank(i), A.size_col(i), A.rank(i+1)*x.rank(i+1), p]);
            else
                Xi = matricize( x.U{i}, 2);
                V{i} = A.U{i}*Xi;
                V{i} = reshape( V{i}, [A.rank(i), A.rank(i+1), A.size_col(i), x.rank(i), x.rank(i+1)] );
                V{i} = permute( V{i}, [1, 4, 3, 2, 5]);
                V{i} = reshape( V{i}, [A.rank(i)*x.rank(i), A.size_col(i), A.rank(i+1)*x.rank(i+1)]);
            end
        end
        y = TTeMPS( V );
    else
        %check for possible block format
        p = size(x,4);
        if p ~= 1
            Xi = permute( x, [2 1 3 4]);
            Xi = reshape( Xi, [A.size_row(idx), size(x, 1)*size(x, 3)*p] );
            V = A.U{idx}*Xi;
            V = reshape( V, [A.rank(idx), A.rank(idx+1), A.size_col(idx), size(x, 1), size(x, 3), p] );
            V = permute( V, [1, 4, 3, 2, 5, 6]);
            y = reshape( V, [A.rank(idx)*size(x, 1), A.size_col(idx), A.rank(idx+1)*size(x, 3), p]);
        else
            Xi = matricize( x, 2);
            V = A.U{idx}*Xi;
            V = reshape( V, [A.rank(idx), A.rank(idx+1), A.size_col(idx), size(x, 1), size(x, 3)] );
            V = permute( V, [1, 4, 3, 2, 5]);
            y = reshape( V, [A.rank(idx)*size(x,1), A.size_col(idx), A.rank(idx+1)*size(x,3)]);
        end
    end

end
