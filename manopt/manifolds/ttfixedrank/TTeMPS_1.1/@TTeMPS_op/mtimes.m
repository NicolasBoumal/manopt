function A = mtimes( B, A )
    %MINUS Multiplication of TT/MPS operator by scalar.
    %   A = MTIMES(B, A) multiplies the TT/MPS operator A by the scalar B.
    %
    %   See also PLUS.
    
    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    %A.U{1} = B*A.U{1};
    %Numerically more stable way: distribute scalar over all cores:

    b = B^(1/A.order);
    for i = 1:A.order
        A.U{i} = b*A.U{i};
    end
end
