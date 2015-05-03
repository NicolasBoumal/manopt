function Y = multiher(X)
% Returns the Hermitian parts of the matrices in the 3D matrix X
%
% function Y = multiher(X)
%
% Y is a 3D matrix the same size as X. Each slice Y(:, :, i) is the
% Hermitian part of the slice X(:, :, i).
%
% See also: multiprod multitransp multiscale multiskew

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Jan. 31, 2013.
% Contributors: 
% Change log: 

    Y = .5*(X + multihconj(X));
    
end