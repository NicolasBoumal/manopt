function lift = ballslift(n, m)
% Projection lift from a sphere in R^(n+1) to a ball in R^n, m times.
%
% function lift = ballslift(n)
% function lift = ballslift(n, m)
%
% This function produces a lift structure to be used with manoptlift.
%
% If m is omitted, then m = 1 by default.
%
% The upstairs manifold M is a product of m spheres in R^(n+1).
% The downstairs manifold N consist of real matrices of size n x m.
%
% Let Y be a point upstairs, so that it is a matrix of size (n+1) x m.
% The lift is
% 
%   phi(Y) = Y(1:n, :),
% 
% that is, it projects one dimension down by removing the last coordinate.
%
% The image of the lift downstairs is phi(M) = B^m, where B is a ball in
% R^n, that is:
%
%   B = {x in R^n : norm(x) <= 1}
%
% This lift can be used in Manopt to optimize over m points in a ball.
%
% See https://arxiv.org/abs/2207.03512, Example 4.3 and Corollary 4.12 for
% theoretical properties of this lift, e.g., the fact that second-order
% critical points for the problem upstairs map to first-order stationary
% points for the problem downstairs, and also that local minima upstairs
% map to local minima downstairs.
%
% See also: manoptlift cubeslift

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, June 21, 2024.
% Contributors: 
% Change log: 

    if ~exist('m', 'var') || isempty(m)
        m = 1;
    end

    if m == 1
        lift.M = spherefactory(n+1);
    elseif m > 1
        lift.M = obliquefactory(n+1, m);
    else
        error('m should be a positive integer (it is 1 by default).');
    end
    lift.N = euclideanfactory(n, m);

    % TODO: The code below works for any upstairs manifold that is a
    % Riemannian submanifold of euclideanfactory(n+1, m).
    % Make the function more general by allowing to pass an argument?
    % What would be other useful examples?

    lift.embedded = true;
    lift.phi = @(Y) Y(1:n, :);
    lift.Dphi = @(Y, V) V(1:n, :);
    lift.Dphit = @(Y, U) [U ; zeros(1, m)];
    lift.hesshw = @(Y, V, W) zeros(n+1, m);

end
