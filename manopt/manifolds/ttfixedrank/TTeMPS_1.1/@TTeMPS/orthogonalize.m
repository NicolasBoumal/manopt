function x = orthogonalize( x, pos )
    %ORTHOGONALIZE Orthogonalize tensor.
    %   X = ORTHOGONALIZE( X, POS ) orthogonalizes all cores of the TTeMPS tensor X
    %   except the core at position POS. Cores 1...POS-1 are left-, cores POS+1...end
    %   are right-orthogonalized. Therefore,
    %
    %   X = ORTHOGONALIZE( X, 1 ) right-orthogonalizes the full tensor,
    %
    %   X = ORTHOGONALIZE( X, X.order ) left-orthogonalizes the full tensor.
    %
    %   See also ORTH_AT.
    
    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2014
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    % left orthogonalization till pos (from left)
    for i = 1:pos-1
        x = orth_at( x, i, 'left' );
    end

    % right orthogonalization till pos (from right)
    for i = x.order:-1:pos+1
        x = orth_at( x, i, 'right' );
    end
end
