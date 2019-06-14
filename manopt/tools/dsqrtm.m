function [D, fX] = dsqrtm(X, H)
% Fréchet derivative of the matrix square root.
%
% function [D, fX] = dsqrtm(X, H)
%
% Computes the directional derivative (the Fréchet derivative) of sqrtm at
% X along H (square matrices).
%
% Thus, D = lim_(t -> 0) (sqrtm(X + tH) - sqrtm(X)) / t.
%
% The second output is fX = sqrtm(X), though it may be less accurate.
% 
% See also: dfunm dlogm dexpm

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, July 3, 2015.
% Contributors:
% Change log:
%
%   June 14, 2019 (NB): now also outputs sqrtm(X) as a by-product.
    
    [D, fX] = dfunm(@sqrtm, X, H);
    
end
