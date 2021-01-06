function x = round( x, r )
    %ROUND Approximate TTeMPS tensor within a prescribed tolerance.
    %   X = ROUND( X, tol ) truncates the given TTeMPS tensor X to a
    %   lower rank such that the error is in order of tol.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt
    
    y = TTeMPS_block_to_TTeMPS( x );
    y = truncate(y , r);
    x = TTeMPS_block.TTeMPS_to_TTeMPS_block( y, x.mu, x.p );

end
