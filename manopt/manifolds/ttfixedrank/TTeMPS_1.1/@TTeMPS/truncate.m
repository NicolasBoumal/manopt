function x = truncate( x, r )
    %TRUNCATE Truncate TTeMPS tensor to prescribed rank.
    %   X = TRUNCATE( X, R ) truncates the given TTeMPS tensor X to rank R. 
    % The result is a right orthogonal TTeMPS.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt
    

    % Left-right procedure
    x = x.orthogonalize( x.order );

    for i = x.order:-1:2
        [U,S,V] = svd( unfold( x.U{i}, 'right'), 'econ');
        s = min( r(i), length(S));
        U = U(:,1:s);
        V = V(:,1:s);
        S = S(1:s,1:s);
        x.U{i} = reshape( V', [s, x.size(i), x.rank(i+1)] );
        x.U{i-1} = tensorprod( x.U{i-1}, (U*S)', 3 );
    end

end
