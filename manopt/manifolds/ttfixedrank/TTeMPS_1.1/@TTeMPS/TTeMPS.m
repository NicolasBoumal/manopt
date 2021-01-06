classdef TTeMPS
% TTeMPS
%
%   A MATLAB class for representing and manipulating tensors
%   in the TT/MPS format. 
% 

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

properties( SetAccess = public, GetAccess = public )

    U           % core tensors

end

% Dependent properties
properties( Dependent = true, SetAccess = private, GetAccess = public )

    rank
    order
    size

end

% Get methods for dependent properties (computed on-the-fly)
methods

    function rank_ = get.rank(x)
        rank_ = cellfun( @(x) size(x,1), x.U);
        rank_ = [rank_, size(x.U{end},3)];
    end
   
    function size_ = get.size(x)
        size_ = cellfun( @(y) size(y,2), x.U);
    end

    function order_ = get.order(x)
        order_ = length( x.U );
    end
end


methods( Access = public )

    function x = TTeMPS(varargin)
    %TTEMPS Construct a tensor in TT/MPS format and return TTeMPS object.
    %
    %   X = TTEMPS() creates a zero TT/MPS tensor 
    %
    %   X = TTEMPS(CORES) creates a TT/MPS tensor with core tensors C taken 
    %   from the cell array CORES
    %

        % Default constructor
        if (nargin == 0)
            x = TTeMPS( {0 0 0} );
            return;
        elseif (nargin == 1)
            x.U = varargin{1};
            return;
        else
            error('Invalid number of arguments.')
        end
    end

    % Other public functions
    y = full( x );
    [x, r] = orth_at( x, pos, dir, apply );
    x = orthogonalize( x, pos );
    res = innerprod( x, y, dir, upto, storeParts );
    res = norm( x, safe );
    res = contract( x, y, idx );
    x = truncate( x, r );
    x = uminus( x );
    x = uplus( x );
    z = plus( x, y );
    z = minus( x, y );
    x = mtimes( a, x );
    z = hadamard( x, y, idx );
    y = mergecores( x, idx );
    y = splitcore( x, idx, nL, nR, tol );
    z = hadamard_division( x, y, idx );
    z = TTeMPS_to_TT( x );
    z = cat( mu, x, y );
    [xl,xr,g] = gauge_matrices( x );
    disp( x, name );
    display( x );
    
end

methods( Static, Access = public )
    x = from_array( A, opts );
end

methods( Static, Access = private )

    x = subsref_mex( r, n, ind , C);

end
end
