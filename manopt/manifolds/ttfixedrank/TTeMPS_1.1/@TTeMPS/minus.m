function z = minus( x, y )
    %MINUS Substraction of two TT/MPS tensors.
    %   Z = MINUS(X,Y) substracts two TT/MPS tensors. The rank of the resulting
    %   tensor is the sum of the individual ranks.
    %
    %   See also PLUS, UMINUS.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    z = plus(x, uminus(y));

end
