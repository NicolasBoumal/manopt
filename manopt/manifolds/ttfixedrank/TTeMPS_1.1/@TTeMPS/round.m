function x = round( x, tol )
    %ROUND Approximate TTeMPS tensor within a prescribed tolerance.
    %   X = ROUND( X, tol ) truncates the given TTeMPS tensor X to a
    %   lower rank such that the error is in order of tol.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt
    
    sz = x.size;
    d = x.order;
    
    % Left-right procedure
    x = x.orthogonalize( d );

    right_rank = 1;
    for i = d:-1:2
        [U,S,V] = svd( unfold( x.U{i}, 'right'), 'econ' );
        r = trunc_singular( diag(S), tol, true );
        U = U(:,1:r);
        V = V(:,1:r);
        S = S(1:r,1:r);
        x.U{i} = reshape( V', [r, sz(i), right_rank] );
        x.U{i-1} = tensorprod( x.U{i-1}, (U*S).', 3 );
        right_rank = r;
    end

end
