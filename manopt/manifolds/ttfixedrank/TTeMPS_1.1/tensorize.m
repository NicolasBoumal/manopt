function res = tensorize( U, mode, d )
    %TENSORIZE Tensorize matrix (inverse matricization).
    %   X = TENSORIZE(U, MODE, D) (re-)tensorizes the matrix U along the 
    %   specified mode MODE into a tensor X of size D(1) x D(2) x D(3). Higher 
    %   dimensions than 3 are not supported. Tensorize is inverse matricization,
    %   that is, X == tensorize( matricize(X, i), i, size(X)) for all modes i.
    %
    %   See also MATRICIZE, TENSORPROD_TTEMPS, UNFOLD.
    
    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    % pad with 1 for the last dim (annoying)
    if length(d) == 2
        d = [d, 1];
    end
    
    switch mode
        case 1
            res = reshape( U, d );
        case 2 
            res = ipermute( reshape(U, [d(2), d(1), d(3)]), [2, 1, 3] );
        case 3 
            res = reshape( transpose(U), d );
        otherwise
            error('Invalid mode input in function matricize')
    end
end
