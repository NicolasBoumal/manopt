function Y = multiherm(X)
% Returns the Hermitian parts of the matrices in a 3D array
%
% function Y = multiherm(X)
%
% Y is a 3D array the same size as X. Each slice Y(:, :, i) is the
% Hermitian part of the slice X(:, :, i), that is,
%
%   Y(:, :, i) = .5*(X(:, :, i) + X(:, :, i)')
%
% See also: multisym multiskew multiskewh multihconj multiprod multitransp multiscale

% This file is part of Manopt: www.manopt.org.
% Original author: Hiroyuki Sato, April 27, 2015.
% Contributors: 
% Change log: 

    Y = .5*(X + multihconj(X));
    
end
