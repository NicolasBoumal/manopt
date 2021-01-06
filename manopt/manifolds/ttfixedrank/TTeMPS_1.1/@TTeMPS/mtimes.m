function x = mtimes( a, x )
    %MINUS Multiplication of TT/MPS tensor by scalar.
    %   X = MTIMES(A, X) multiplies the TT/MPS tensor X by the scalar A.
    %
    %   See also PLUS, MINUS.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    %x.U{1} = a*x.U{1};

    c = a^(1/x.order);
    for i = 1:x.order
        x.U{i} = c*x.U{i};
    end
end
