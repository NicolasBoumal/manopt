function sqnorm = multisqnorm(A)
% Returns the squared Frobenius norms of the slices of a 3D matrix.
%
% function sqnorm = multisqnorm(A)
%
% Given a 3-dimensional matrix A of size n-by-m-by-N, returns a column
% vector of length N such that sqnorm(i) = norm(A(:, :, i), 'fro')^2.
%
% See also: multiprod multitransp multitrace

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, June 17, 2015.
% Contributors: 
% Change log: 


	assert(ndims(A) <= 3, ...
           ['multisqnorm is only well defined for matrix arrays of 3 ' ...
            'or less dimensions.']);
	[n, m, N] = size(A);
    
    sqnorm = sum(reshape(A, n*m, N).^2, 1)';

end
