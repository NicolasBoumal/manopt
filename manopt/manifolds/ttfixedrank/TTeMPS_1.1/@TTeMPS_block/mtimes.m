function x = mtimes( a, x )
    %MINUS Multiplication of TT/MPS block tensor by scalar or vector
    %   X = MTIMES(A, X) multiplies the TT/MPS tensor X by A.
    %       If A is a scalar, all blocks are multiplied by the x.order-th root of A.
    %       if A is a vector of size X.p, all X.p slices of the supercore X.U{x.mu} are multiplied
    %       by the corresponding entry in A.
    %
    %   See also PLUS, MINUS.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    %x.U{1} = a*x.U{1};

    if length(a) == 1
        c = a^(1/x.order);
        for i = 1:x.order
            x.U{i} = c*x.U{i};
        end
    elseif length(a) == x.p
        for i = 1:x.p
            x.U{x.mu}(:,:,:,i) = a(i) * x.U{x.mu}(:,:,:,i);
        end
    else
        error('Dimension mismatch! Can only multiply block tensor X by scalar (whole tensor) or by a vector of length X.p')
    end
end
