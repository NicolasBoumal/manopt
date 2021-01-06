%LAPLACE Construct a simple d-dimensional Laplace operator and return TTeMPS_op_laplace object.

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

function L = laplace( I, n, d );

    one = ones(n,1);
    q = linspace( I(1), I(2), n)';
    h = abs(q(2) - q(1));
    L0 = -spdiags( [one, -2*one, one], [-1 0 1], n, n) / (h^2);
    
    L = TTeMPS_op_laplace( L0, d );
end
