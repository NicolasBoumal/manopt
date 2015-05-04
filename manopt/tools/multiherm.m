function Y = multiherm(X)
% Returns the Hermitian parts of the matrices in the 3D matrix X
%
% function Y = multiherm(X)
%
% Y is a 3D matrix the same size as X. Each slice Y(:, :, i) is the
% Hermitian part of the slice X(:, :, i).
%
% See also: multiprod multitransp multihconj multiscale multiskew

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, May 4, 2015.
% Contributors: 
% Change log: 

    Y = .5*(X + multihconj(X));
    
end
