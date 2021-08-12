function Y = multisym(X)
% Returns the symmetric parts of the matrices in a 3D array
%
% function Y = multisym(X)
%
% Y is a 3D array the same size as X. Each slice Y(:, :, i) is the
% symmetric part of the slice X(:, :, i), that is,
%
%   Y(:, :, i) = .5*(X(:, :, i) + X(:, :, i).')
%
% Note that we do not take complex conjugates. For this, see multiherm.
%
% See also: multiherm multiskew multiskewh multiprod multitransp multiscale

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Jan. 31, 2013.
% Contributors: 
% Change log: 

    Y = .5*(X + multitransp(X));
    
end
