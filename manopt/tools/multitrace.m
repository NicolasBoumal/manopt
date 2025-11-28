function tr = multitrace(A)
% Computes the traces of the 2D slices in a 3D array.
% 
% function tr = multitrace(A)
%
% For a 3-dimensional array A of size n-by-m-by-N, returns a column vector
% tr of length N such that tr(k) = sum(diag(A(:, :, k)). In particular,
% if n = m (each slice is square), then tr(k) = trace(A(:, :, k)).
%
% See also: multiprod multitransp multiscale

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
%
%   Nov. 28, 2025 (NB):
%       Removed call to diagsum in favor of more direct code, and now
%       allowing non-square slices just in case.

    
    assert(ndims(A) <= 3, ...
           ['multitrace is only well defined for arrays of 3 or ' ...
            'fewer dimensions.']);

    [n, m, N] = size(A);

    B = reshape(A, n*m, N);

    tr = sum(B(1:(n+1):end, :), 1).';
    
end
