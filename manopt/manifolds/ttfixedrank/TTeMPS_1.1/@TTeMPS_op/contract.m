function res = contract( A, X, mu )
    %CONTRACT Contraction of two TT/MPS tensors with inner TT/MPS operator.
    %   Z = CONTRACT(A,X,IDX) contracts all coress of the two TT/MPS tensors X and Y=apply(A*X) except 
    %   core IDX. Result Z is a matrix of size 
    %       [ X.rank(IDX)*A.size_col(IDX)*X.rank(IDX+1), X.rank(IDX)*A.size_row(IDX),X.rank(IDX+1) ].
    %
    %   See also APPLY.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    V = cell(1, A.order);

    Y = A.apply(X);
    if mu > 1
        left = innerprod( X, Y, 'LR', mu-1 );
        left = reshape( left, [X.rank(mu), A.rank(mu), X.rank(mu)] );
        left = reshape( permute(left, [1 3 2]), [X.rank(mu)*X.rank(mu), A.rank(mu)] );
        res = left * reshape(A.U{mu}, [A.rank(mu), A.size_col(mu)*A.size_row(mu)*A.rank(mu+1)] );
    else
        res = A.U{1};
    end
    if mu < X.order
        right = innerprod( X, Y, 'RL', mu+1 );
        right = reshape( right, [X.rank(mu+1), A.rank(mu+1), X.rank(mu+1)] );
        right = matricize( right, 2);
        res = reshape( res, [ X.rank(mu)*X.rank(mu)*A.size_col(mu)*A.size_row(mu), A.rank(mu+1) ]);
        res = res * right; 
    end

    res = reshape( res, [X.rank(mu), X.rank(mu), A.size_col(mu), A.size_row(mu), X.rank(mu+1), X.rank(mu+1) ] );
    res = permute( res, [1 3 5 2 4 6] );
    res = reshape( res, [X.rank(mu)*A.size_col(mu)*X.rank(mu+1), X.rank(mu)*A.size_row(mu)*X.rank(mu+1) ] );

end
    
