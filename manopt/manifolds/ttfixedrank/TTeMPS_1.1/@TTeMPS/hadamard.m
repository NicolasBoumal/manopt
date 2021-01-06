function z = hadamard( x, y, idx )
    %HADAMARD Hadamard product of two TT/MPS tensors.
    %   Z = HADAMARD(X, Y) calculates the Hadamard product of two TT/MPS 
    %   tensors X and Y and returns the new TT/MPS tensor Z.
    %
    %   Z = HADAMARD(X, U, idx) calculates the Hadamard product of the TT/MPS 
    %   tensors X and the 3D Matlab array Y, assumed to be in core position IDX.
    %   Only core IDX of the resulting tensor is returned.
    %
    %   See also MTIMES, INNERPROD, PLUS, MINUS.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    rx = x.rank;
    n = x.size;
    d = x.order;

    % first case: all cores.
    if ~exist( 'idx', 'var' )
        ry = y.rank;

        C = cell(1,d);
        for i = 1:d
            % move outer dimension to the front
            xx1 = permute( x.U{i}, [2 1 3]);
            p = size( y.U{i}, 4 );
            C{i} = zeros( [rx(i)*ry(i), n(i), rx(i+1)*ry(i+1), p] );
            for j = 1:p
                yy = permute( y.U{i}(:,:,:,j), [2 1 3]);

                % duplicate entries
                xx = repmat( xx1(:), [1, ry(i)*ry(i+1)]);
                yy = repmat( yy(:), [1, rx(i)*rx(i+1)]);

                % reshape and permute to same sizes
                xx = reshape( xx, [n(i), rx(i), rx(i+1), ry(i), ry(i+1)]);
                yy = reshape( yy, [n(i), ry(i), ry(i+1), rx(i), rx(i+1)]);

                xx = permute( xx, [1 2 4 3 5]);
                yy = permute( yy, [1 4 2 5 3]);

                xx = reshape( xx, [n(i), rx(i)*ry(i), rx(i+1)*ry(i+1)]);
                yy = reshape( yy, [n(i), rx(i)*ry(i), rx(i+1)*ry(i+1)]);

                % multiply elementwise.
                zz = xx.*yy;
                C{i}(:,:,:,j) = permute( zz, [2 1 3] );
            end
        end
        z = TTeMPS( C );
    else
        i = idx;
        ry = [size(y,1), size(y,3)];

        % move outer dimension to the front
        xx1 = permute( x.U{i}, [2 1 3]);
        p = size( y, 4 );
        z = zeros( [rx(i)*ry(1), n(i), rx(i+1)*ry(2), p] );
        for j = 1:p
            yy = permute( y(:,:,:,j), [2 1 3]);

            % duplicate entries
            xx = repmat( xx1(:), [1, ry(1)*ry(2)]);
            yy = repmat( yy(:), [1, rx(i)*rx(i+1)]);

            % reshape and permute to same sizes
            xx = reshape( xx, [n(i), rx(i), rx(i+1), ry(1), ry(2)]);
            yy = reshape( yy, [n(i), ry(1), ry(2), rx(i), rx(i+1)]);

            xx = permute( xx, [1 2 4 3 5]);
            yy = permute( yy, [1 4 2 5 3]);

            xx = reshape( xx, [n(i), rx(i)*ry(1), rx(i+1)*ry(2)]);
            yy = reshape( yy, [n(i), rx(i)*ry(1), rx(i+1)*ry(2)]);

            % multiply elementwise.
            zz = xx.*yy;
            z(:,:,:,j) = permute( zz, [2 1 3] );
        end
    end

end
