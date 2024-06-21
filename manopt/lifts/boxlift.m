function lift = boxlift(n, m)
% Sine lift y -> sin(y) for optimization in a box [-1, 1]^n with Manopt.
%
% function lift = boxlift(n)
% function lift = boxlift(n, m)
%
% This function produces a lift structure to be used with manoptlift.
%
% If m is omitted, then m = 1 by default.
%
% Both the upstairs manifold M and the downstairs manifold N consist of
% real matrices of size n x m, that is,
% 
%   lift.M = lift.N = euclideanfactory(n, m).
%
% The lift is phi(y) = sin(y).
%
% The image of the lift downstairs is phi(M) = [-1, 1]^(n x m): a box.
%
% See also: manoptlift

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, June 20, 2024.
% Contributors: 
% Change log: 

    % TODO: this is called "boxlift" and not "cubelift" because it would
    % make sense to allow users to specify lower-bounds and upper-bounds
    % for each entry, and to lift the corresponding right parallelepiped.

    if ~exist('m', 'var') || isempty(m)
        m = 1;
    end

    lift.M = euclideanfactory(n, m);
    lift.N = euclideanfactory(n, m);

    lift.embedded = true;
    lift.phi = @(y) sin(y);
    lift.Dphi = @(y, v) cos(y).*v;
    lift.Dphit = @(y, u) cos(y).*u;
    lift.hesshw = @(y, v, w) -sin(y).*w.*v;

end
