function x = uminus( x )
    %UMINUS Unary minus.
    %   X = UMINUS(X) returns the negated TTeMPS block-mu tensor X. Only the supercore 
    %   is touched.
    %
    %   See also UPLUS.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

        
    x.U{x.mu} = -x.U{x.mu};
end
