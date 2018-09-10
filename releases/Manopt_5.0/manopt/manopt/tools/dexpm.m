function D = dexpm(X, H)
% Fréchet derivative of the matrix exponential.
%
% function D = dexpm(X, H)
%
% Computes the directional derivative (the Fréchet derivative) of expm at X
% along H (square matrices).
%
% Thus, D = lim_(t -> 0) (expm(X + tH) - expm(X)) / t.
%
% Note: the adjoint of dexpm(X, .) is dexpm(X', .), which is a fact often
% useful to derive gradients of matrix functions involving expm(X).
% (This is wrt the inner product inner = @(A, B) real(trace(A'*B))).
% 
% See also: dfunm dlogm dsqrtm

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, July 3, 2015.
% Contributors:
% Change log:
    
    D = dfunm(@expm, X, H);
    
end
