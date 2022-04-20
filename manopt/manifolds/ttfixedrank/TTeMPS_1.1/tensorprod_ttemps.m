function res = tensorprod_ttemps( U, A, mode, apply_inv )
    %TENSORPROD_TTEMPS Tensor-times-Matrix product. 
    %   A = TENSORPROD_TTEMPS(U, A, MODE) performs the mode-MODE product between the
    %   tensor U and matrix A. Higher dimensions than 3 are not supported.
    %
    %   A = TENSORPROD_TTEMPS(U, A, MODE, TRUE) multiplies with A^{-1} instead of A.
	%
	%   Renamed from tensorprod to tensorprod on April 20, 2022 to accomodate the new
	%   Matlab built-in function tensorprod in R2022a.
    %
    %   See also MATRICIZE, TENSORIZE, UNFOLD.
    
    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    if nargin==3
        apply_inv = false;
    end
    
    d = size(U);
    % pad with 1 for the last dim (annoying)
    if length(d) == 2
        d = [d, 1];
    end
    d(mode) = size(A,1);
    
    if apply_inv
        res = A \ matricize( U, mode );
        res = tensorize( res, mode, d );
    else
        res = A * matricize( U, mode );
        res = tensorize( res, mode, d );
    end

end
