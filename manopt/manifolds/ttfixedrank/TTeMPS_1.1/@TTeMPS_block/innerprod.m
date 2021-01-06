function res = innerprod( x, y, dir, upto )
    %INNERPROD Inner product between two TT/MPS tensors.
    %   innerprod(X,Y) computes the inner product between the TT/MPS tensors X and Y.
    %   Assumes that the first rank of both tensors, X.rank(1) and Y.rank(1), is 1. 
    %   The last rank may be different from 1, resulting in a matrix of size 
    %   [X.rank(end), Y.rank(end)].
    %
    %   See also NORM

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    if ~exist( 'dir', 'var' )
        dir = 'LR';
    end
    if ~exist( 'upto', 'var' )
        if strcmpi( dir, 'LR')
            upto = x.order;
        else
            upto = 1;
        end
    end

    % Left-to-Right procedure
    if strcmpi( dir, 'LR')
        
        res = unfold( x.U{1}, 'left')' * unfold( y.U{1}, 'left');

        for i = 2:upto
            tmp = tensorprod( x.U{i}, res', 1);
            res = unfold( tmp, 'left')' * unfold( y.U{i}, 'left');
        end

    % Right-to-Left procedure
    elseif strcmpi( dir, 'RL')
        d = x.order;
        res = conj(unfold( x.U{d}, 'right')) * unfold( y.U{d}, 'right').';
        
        for i = d-1:-1:upto
            tmp = tensorprod( x.U{i}, res', 3);
            res = conj(unfold( tmp, 'right')) * unfold( y.U{i}, 'right').';
        end

    else
        error('Unknown direction specified. Choose either LR (default) or RL')
    end

end

