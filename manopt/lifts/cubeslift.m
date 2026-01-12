function lift = cubeslift(n, m, scale)
% Sine lift: y -> sin(y) for optimization in a box [-1, 1]^n with Manopt.
%
% function lift = cubeslift(n)
% function lift = cubeslift(n, m)
% function lift = cubeslift(n, m, scale)
%
% This function produces a lift structure to be used with manoptlift.
%
% If m is omitted, then m = 1 by default.
% If scale is omitted, then scale = 1 by default.
%
% Both the upstairs manifold M and the downstairs manifold N consist of
% real matrices of size n x m, that is,
% 
%   lift.M = lift.N = euclideanfactory(n, m).
%
% The lift is phi(y) = scale*sin(y).
%
% The image of the lift downstairs is a box:
% 
%   phi(M) = [-scale, scale]^(n x m).
%
% See also: manoptlift

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, June 20, 2024.
% Contributors: 
% Change log: 
%   Jan. 12, 2026 (NB): added 'scale' as an input.

    if ~exist('m', 'var') || isempty(m)
        m = 1;
    end

    if ~exist('scale', 'var') || isempty(scale)
        scale = 1;
    end

    lift.M = euclideanfactory(n, m);
    lift.N = euclideanfactory(n, m);

    lift.embedded = false;
    lift.phi = @(y) scale*sin(y);
    lift.Dphi = @(y, v) scale*cos(y).*v;
    lift.Dphit = @(y, u) scale*cos(y).*u;
    lift.hesshw = @(y, v, w) -scale*sin(y).*w.*v;

end
