classdef TTeMPS_block
% TTeMPS_block
%
%   A MATLAB class for representing and manipulating tensors
%   in the TT/MPS Block-mu format. 
% 

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

properties( SetAccess = public, GetAccess = public )

    U           % core tensors
    mu          % position of superblock
    p           % size of block
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
        order_ = length( x.size );
    end
end


methods( Access = public )

    function x = TTeMPS_block(varargin)
    %TTEMPS_BLOCK Construct a tensor in TT/MPS block-mu format and return TTeMPS_block object.
    %
    %   X = TTEMPS_BLOCK() creates a zero TT/MPS tensor 
    %
    %   X = TTEMPS_BLOCK(CORES) creates a TT/MPS tensor with core tensors C taken 
    %   from the cell array CORES
    %
    %   X = TTEMPS_BLOCK(CORES, ORTH) creates a TT/MPS tensor with core tensors C 
    %   taken from the cell array CORES. ORTH specifies the position of 
    %   orthogonalization (default = 0, no orthogonalization).
    %

        % Default constructor
        if (nargin == 0)
          
            x = TTeMPS_block( {0 0 0}, 1 );
            return;
          
        elseif (nargin == 2)

            % CAREFUL, add sanity check here
            U = varargin{1};
            mu = varargin{2}; 
            x = TTeMPS_block( U, mu, size(U{mu},4) );
            return;

        elseif (nargin == 3)

            x.U = varargin{1};
            x.mu = varargin{2};
            x.p = varargin{3};

        else
            error('Invalid number of arguments.')
        end
    end

    function res = TTeMPS_block_to_TTeMPS( x )
        mu = x.mu;
        tmp = permute( x.U{mu}, [1 2 4 3] );
        tmp = reshape( tmp, [x.rank(mu), x.size(mu)*x.p, x.rank(mu+1)]);

        res = TTeMPS({x.U{1:mu-1}, tmp, x.U{mu+1:end}});
    end

    function res = getVector( x, idx, condense )
        if ~exist( 'condense', 'var' )
            condense = false;
        end
        tmp = x.U{x.mu}(:,:,:,idx);
        res = TTeMPS({x.U{1:x.mu-1}, tmp, x.U{x.mu+1:end}});
        if condense
            res = round(res, 1e-16);
        end
    end


    % Other public functions
    y = full( x );
    [x, r] = orth_at( x, pos, dir, apply );
    res = innerprod( x, y, dir, upto, storeParts );
    res = shift( x, nu, tol, maxrank);
    res = norm( x );
    res = contract( x, y, idx );
    x = round( x, tol );
    x = truncate( x, r );
    x = uminus( x );
    x = uplus( x );
    z = plus( x, y );
    z = minus( x, y );
    x = mtimes( a, x );
    z = hadamard( x, y, idx );
    [xl,xr,g] = gauge_matrices( x );
    disp( x, name );
    display( x );
    
end

methods( Static, Access = public )

    function res = TTeMPS_to_TTeMPS_block( x, mu, p )

        U = x.U;
        tmp = reshape( U{mu}, [x.rank(mu), x.size(mu)/p, p, x.rank(mu+1)]);
        tmp = permute( tmp, [1 2 4 3] );
        res = TTeMPS_block({U{1:mu-1}, tmp, U{mu+1:x.order}}, mu, p);
    end

end

end
