function [x, R] = orth_at( x, pos, dir, apply )
    %ORTH_AT Orthogonalize single core.
    %   X = ORTH_AT( X, POS, 'LEFT') left-orthogonalizes the core at position POS 
    %   and multiplies the corresponding R-factor with core POS+1. All other cores
    %   are untouched. The modified tensor is returned.
    %
    %   X = ORTH_AT( X, POS, 'RIGHT') right-orthogonalizes the core at position POS
    %   and multiplies the corresponding R-factor with core POS-1. All other cores
    %   are untouched. The modified tensor is returned.
    %
    %   See also ORTHOGONALIZE.
    
    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    if ~exist( 'apply', 'var' )
        apply = true;
    end

    sz = size(x.U{pos});
    if length(sz) == 2 
        sz = [sz, 1];  
    end

    if strcmpi(dir, 'left')
        [Q,R] = qr( unfold( x.U{pos}, 'left' ), 0);
        x.U{pos} = reshape( Q, [sz(1), sz(2), size(Q,2)] );
        if apply
            x.U{pos+1} = tensorprod( x.U{pos+1}, R, 1); 
        end

    elseif strcmpi(dir, 'right') 
        % mind the transpose as we want to orthonormalize rows
        [Q,R] = qr( unfold( x.U{pos}, 'right' )', 0);
        x.U{pos} = reshape( Q', [size(Q,2), sz(2), sz(3)] );
        if apply
            x.U{pos-1} = tensorprod( x.U{pos-1}, R, 3); 
        end
        
    else
        error('Unknown direction specified. Choose either LEFT or RIGHT') 
    end
end
