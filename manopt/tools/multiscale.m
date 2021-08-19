function A = multiscale(scale, A)
% Multiplies the 2D slices in a 3D matrix by individual scalars.
%
% function A = multiscale(scale, A)
%
% Given a vector scale of length N and a 3-D array A of size
% n-by-m-by-N, returns an array B of same size as A such that
% B(:, :, k) = scale(k) * A(:, :, k);
%
% See also: multiprod multitransp multitrace

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 


    assert(ndims(A) <= 3, ...
           ['multiscale is only well defined for matrix arrays of 3 ' ...
            'or fewer dimensions.']);
    [n, m, N] = size(A);
    assert(numel(scale) == N, ...
           ['scale must be a vector whose length equals the third ' ...
            'dimension of A, that is, the number of 2D matrix slices ' ...
            'in the 3D array A.']);

    scale = scale(:);
    A = reshape(bsxfun(@times, reshape(A, n*m, N), scale'), n, m, N);

end
