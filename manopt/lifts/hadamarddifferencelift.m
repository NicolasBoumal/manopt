function lift = hadamarddifferencelift(n)
% Lift phi(u, v) = u.^2 - v.^2 for R^n, with 1-norm regularizer.
%
% function lift = hadamarddifferencelift(n)
%
% This function produces a lift structure to be used with manoptlift.
% 
% The upstairs manifold M consists of real matrices Y of size n-by-2,
% thought of as two vectors: Y = [u, v].
%
% The downstairs manifold N is R^n.
%
% The lift is phi(u, v) = u.*u - v.*v. Its image is all of R^n.
%   
% The built-in regularizer is
%
%   rho(u, v) = ||u||^2 + ||v||^2    (in 2-norms).
%
% It can be activated with manoptlift, using the lambda parameter.
% Using this regularizer upstairs amounts to 1-norm regularization
% downstairs because
%
%    r(x)  =  min_{u, v : phi(u, v) = x} rho(u, v)  =  norm(x, 1).
%
% This promotes sparsity. For more on this parameterization, see:
%  https://proceedings.neurips.cc/paper/2019/hash/5cf21ce30208cfffaa832c6e44bb567d-Abstract.html
%  https://proceedings.mlr.press/v125/woodworth20a.html
%  https://arxiv.org/abs/2307.03571
% 
% See also: manoptlift burermonteiroLRlift

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, July 1, 2024.
% Contributors: 
% Change log: 

    lift.M = euclideanfactory(n, 2);
    lift.N = euclideanfactory(n, 1);

    lift.embedded = false;

    lift.phi = @(y) y(:, 1).*y(:, 1) - y(:, 2).*y(:, 2);
    lift.Dphi = @(y, v) 2*y(:, 1).*v(:, 1) - 2*y(:, 2).*v(:, 2);
    lift.Dphit = @(y, u) [2*y(:, 1).*u, - 2*y(:, 2).*u];
    lift.hesshw = @(y, v, w) [2*v(:, 1).*w, -2*v(:, 2).*w];

    lift.rho = @(y) norm(y, 'fro')^2;
    lift.gradrho = @(y) 2*y;
    lift.hessrho = @(y, ydot) 2*ydot;

end
