function z = plus( x, y )
    %PLUS Addition of two TT/MPS operators.
    %   Z = PLUS(X,Y) adds to TT/MPS operators. The resulting TT/MPS operator 
    %   has rank equal to the sum of the individual ranks.
    %
    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    
    % add sanity check...
    rx = x.rank;
    ry = y.rank;

    z = TTeMPS_op( cell(1, x.order) );
        
    % first core:
    tmp = zeros( 1, x.size_col(1), x.size_row(1), rx(2)+ry(2) );
    tmp( 1, :, :, 1:rx(2) ) = x.U{1};
    tmp( 1, :, :, rx(2)+1:end ) = y.U{1};
    z.U{1} = tmp;

    %z.U{1} = reshape( [unfold( x.U{1}, 'left'), unfold( y.U{1}, 'left')], [1, x.size(1), x.rank(2) + y.rank(2)]);

    % central cores:
    for i = 2:x.order-1
        tmp = zeros( rx(i)+ry(i), x.size_col(i), x.size_row(i), rx(i+1)+ry(i+1) );
        tmp( 1:rx(i), :, :, 1:rx(i+1) ) = x.U{i};
        tmp( rx(i)+1:end, :, :, rx(i+1)+1:end ) = y.U{i};
        z.U{i} = tmp;
    end

    % last core:
    tmp = zeros( rx(end-1)+ry(end-1), x.size_col(end), x.size_row(end), 1 );
    tmp( 1:rx(end-1), :, :, 1 ) = x.U{end};
    tmp( rx(end-1)+1:end, :, :, 1 ) = y.U{end};
    z.U{end} = tmp;

    z = update_properties( z );
end
