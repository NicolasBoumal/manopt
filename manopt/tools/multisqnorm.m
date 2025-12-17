function sqnorm = multisqnorm(A)
% Returns the squared Frobenius norms of the slices of a 3D array.
%
% function sqnorm = multisqnorm(A)
%
% Given a 3-dimensional array A of size n-by-m-by-N, returns a column
% vector of length N such that sqnorm(i) = norm(A(:, :, i), 'fro')^2.
%
% See also: multiprod multitransp multitrace norms

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, June 17, 2015.
% Contributors: 
% Change log: 
%
%   Nov. 28, 2025 (NB):
%       Now works with complex arrays as well, and is faster.

    assert(ndims(A) <= 3, ...
           ['multisqnorm is only well defined for matrix arrays of 3 ' ...
            'or fewer dimensions.']);

    [n, m, N] = size(A);
    sqnorm = (vecnorm(reshape(A, [n*m, N]), 2, 1).^2).';

    % Below are three essentially equivalent computations but slower:
    % 
    % sqnorm = squeeze(pagenorm(A, 'fro')).^2;
    % 
    % B = reshape(A, [n*m, N]);
    % sqnorm = sum(real(B).^2 + imag(B).^2, 1).';
    % 
    % sqnorm = squeeze(sum(norms(A, 2, 1).^2));

end
