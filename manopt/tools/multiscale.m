function A = multiscale(scale, A)
% Multiplies the 2D slices in a 3D matrix by individual scalars.
%
% function A = multiscale(scale, A)
%
% Given a vector scale of length N and a 3-D array A of size
% n-by-m-by-N, returns an array B of same size as A such that
% B(:, :, k) = scale(k) * A(:, :, k);
%
% See also: multiprod multitransp multitrace cmultiscale

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
%   Aug. 29, 2021 (NB):
%       Corrected bug that occurred for complex 'scale' vector.
%   Sep.  5, 2021 (NB):
%       Using .* rather than bxsfun as a preferred way: this is faster.
%       Kept the bsxfun code in a try/catch in case this causes trouble
%       with older versions of Matlab (unsure whether it would).

    assert(ndims(A) <= 3, ...
           ['multiscale is only defined for arrays of 3 or fewer ' ...
            'dimensions.']);
        
    [n, m, N] = size(A);
    
    assert(numel(scale) == N, ...
           ['scale must be a vector whose length equals the third ' ...
            'dimension of A, that is, the number of 2D matrix slices ' ...
            'in the 3D array A.']);

    try
        A = A .* reshape(scale, [1, 1, N]);
    catch
        scale = scale(:);
        A = reshape(bsxfun(@times, reshape(A, n*m, N), scale.'), n, m, N);
    end

end
