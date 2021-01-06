function x = uminus( x )
    %UMINUS Unary minus.
    %   X = UMINUS(X) returns the negated TTeMPS tensor X. Only the first core 
    %   is touched. 
    %
    %   See also UPLUS.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

        
    x.U{1} = -x.U{1};
end
