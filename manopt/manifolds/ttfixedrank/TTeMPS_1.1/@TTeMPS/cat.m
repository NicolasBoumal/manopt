function z = cat( mu, x, y )
    %CAT concatenation of two TT/MPS tensors.
    %   Z = CAT(MU,X,Y) concatenates two TT/MPS tensors along the
    %   outer dimension MU. 
    %   The rank of the resulting tensor is rank(X) + rank(Y).
    %
    %   See also PLUS, MINUS.
    
    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt
    
    % add sanity check...
    rx = x.rank;
    ry = y.rank;
    nx = x.size;
    ny = y.size;
    

    z = TTeMPS( cell(1, x.order) );
        
    % first core:
    if mu == 1
        p = size(x.U{1},4); %special treatment of block TT tensors
        tmp = zeros( 1, nx(1)+ny(1), rx(2)+ry(2), p );
        tmp( 1, 1:nx(1), 1:rx(2), : ) = x.U{1};
        tmp( 1, nx(1)+1:end, rx(2)+1:end, : ) = y.U{1};
        z.U{1} = tmp;
    else
        p = size(x.U{1},4);
        tmp = zeros( 1, nx(1), rx(2)+ry(2), p );
        tmp( 1, :, 1:rx(2), : ) = x.U{1};
        tmp( 1, :, rx(2)+1:end, : ) = y.U{1};
        z.U{1} = tmp;
    end

    % central cores:
    for i = 2:x.order-1
        if mu == i
            p = size(x.U{i},4);
            tmp = zeros( rx(i)+ry(i), nx(i)+ny(i), rx(i+1)+ry(i+1), p);
            tmp( 1:rx(i), 1:nx(i), 1:rx(i+1), :) = x.U{i};
            tmp( rx(i)+1:end, nx(i)+1:end, rx(i+1)+1:end, :) = y.U{i};
            z.U{i} = tmp;
        else
            % possibility of block format:
            p = size(x.U{i},4);
            tmp = zeros( rx(i)+ry(i), nx(i), rx(i+1)+ry(i+1), p);
            tmp( 1:rx(i), :, 1:rx(i+1), :) = x.U{i};
            tmp( rx(i)+1:end, :, rx(i+1)+1:end, :) = y.U{i};
            z.U{i} = tmp;
        end
    end

    % last core:
    if mu == x.order
        p = size(x.U{end},4);
        tmp = zeros( rx(end-1)+ry(end-1), nx(end)+ny(end), 1, p );
        tmp( 1:rx(end-1), 1:nx(end), 1, : ) = x.U{end};
        tmp( rx(end-1)+1:end, nx(end)+1:end, 1, : ) = y.U{end};
        z.U{end} = tmp;
    else
        p = size(x.U{end},4);
        tmp = zeros( rx(end-1)+ry(end-1), nx(end), 1, p );
        tmp( 1:rx(end-1), :, 1, : ) = x.U{end};
        tmp( rx(end-1)+1:end, :, 1, : ) = y.U{end};
        z.U{end} = tmp;
    end
end
