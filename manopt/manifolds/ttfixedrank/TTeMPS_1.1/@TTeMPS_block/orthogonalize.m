function res = orthogonalize( x )
    %ORTHOGONALIZE Orthogonalize TT/MPS Block-mu tensor.
    %   X = ORTHOGONALIZE( X ) orthogonalizes all cores of the TTeMPS_block tensor X
    %   except the supercore at position X.MU. Cores 1...X.MU-1 are left-, cores X.MU+1...end
    %   are right-orthogonalized. 

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt


    y = TTeMPS_block_to_TTeMPS( x );
    yorth = orthogonalize(y, x.mu );
    res = TTeMPS_block.TTeMPS_to_TTeMPS_block( yorth, x.mu, x.p );

    %tmp = permute( y.U{mu}, [1 2 4 3] );
    %tmp = reshape( tmp, [x.rank(mu), x.size(mu)*x.p, x.rank(mu+1)];

    %X = TTeMPS({x.U{1:mu-1}, tmp, x.U{1:mu+1} );

    

    % left orthogonalization till x.mu (from left)
    %for i = 1:x.mu-1
    %    [Q,R] = qr( unfold( x.U{i}, 'left' ), 0);
    %    x.U{pos} = reshape( Q, [x.rank(pos), x.size(pos), size(Q,2)] );
    %    if apply
    %        x.U{pos+1} = tensorprod( x.U{pos+1}, R, 1); 
    %    end
    %end

    %% right orthogonalization till x.mu (from right)
    %for i = x.order:-1:x.mu+1
    %
    %end

end

