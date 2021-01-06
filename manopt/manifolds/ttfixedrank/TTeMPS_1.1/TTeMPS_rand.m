function x = TTeMPS_rand( r, n )
    %TTEMPS_RAND Create random TTeMPS tensor
    %   X = TTEMPS_RAND( R, N ) creates a length(N)-dimensional TTeMPS tensor
    %   of size N(1)*N(2)*...N(end) with ranks R by filling the the cores with
    %   uniform random numbers.
    %   Note that the first and last entry of the rank vector must be 1.
    %

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt
    
    if length(r) ~= length(n)+1
        error('Size mismatch in arguments')
    end

    U = cell(1, length(n));
    for i=1:length(n)
        U{i} = randn( r(i), n(i), r(i+1) );
    end
    x = TTeMPS( U );
end
