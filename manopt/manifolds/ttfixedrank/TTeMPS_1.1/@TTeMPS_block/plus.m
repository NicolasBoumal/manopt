function z = plus( x, y )
    %PLUS Addition of two TT/MPS block-mu tensors.
    %   Z = PLUS(X,Y) adds two TT/MPS tensors. The rank of the resulting
    %   tensor is 2*R.
    %
    %   See also MINUS, UMINUS.
    
    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt
    
    % add sanity check...

    if ( x.mu ~= y.mu ) || ( x.p ~= y.p )
        error('Summands must have the same TT/MPS block-mu structure!')
    end

    rx = x.rank;
    ry = y.rank;

    z = TTeMPS_block( cell(1, x.order), x.mu, x.p );
        
    % first core:
    if x.mu == 1
        tmp = zeros( 1, x.size(1), rx(2)+ry(2), x.p );
        tmp( 1, :, 1:rx(2), : ) = x.U{1};
        tmp( 1, :, rx(2)+1:end, : ) = y.U{1};
    else
        tmp = zeros( 1, x.size(1), rx(2)+ry(2) );
        tmp( 1, :, 1:rx(2) ) = x.U{1};
        tmp( 1, :, rx(2)+1:end ) = y.U{1};
    end
    z.U{1} = tmp;

    % central cores:
    for i = 2:x.order-1
        % possibility of block format:
        if x.mu == i
            tmp = zeros( rx(i)+ry(i), x.size(i), rx(i+1)+ry(i+1), x.p);
            tmp( 1:rx(i), :, 1:rx(i+1), :) = x.U{i};
            tmp( rx(i)+1:end, :, rx(i+1)+1:end, :) = y.U{i};
        else
            tmp = zeros( rx(i)+ry(i), x.size(i), rx(i+1)+ry(i+1) );
            tmp( 1:rx(i), :, 1:rx(i+1) ) = x.U{i};
            tmp( rx(i)+1:end, :, rx(i+1)+1:end ) = y.U{i};
        end
        z.U{i} = tmp;
    end

    % last core:
    if x.mu == x.order;
        tmp = zeros( rx(end-1)+ry(end-1), x.size(end), 1, x.p );
        tmp( 1:rx(end-1), :, 1, : ) = x.U{end};
        tmp( rx(end-1)+1:end, :, 1, : ) = y.U{end};
    else
        tmp = zeros( rx(end-1)+ry(end-1), x.size(end) );
        tmp( 1:rx(end-1), : ) = x.U{end};
        tmp( rx(end-1)+1:end, : ) = y.U{end};
    end
    z.U{end} = tmp;
end
