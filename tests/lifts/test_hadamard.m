clear; clc; clf;

n = 5;
m = n;
lift = hadamardlift('colstochastic', n, m);

% random stochastic matrix
A = lift.phi(lift.M.rand());

inner = @(U, V) U(:).'*V(:);
sqfrobnorm = @(U) inner(U, U);
downstairs.cost = @(X) sqfrobnorm(X*X-A);

[upstairs, downstairs] = manoptlift(downstairs, lift, 'AD');

Y = trustregions(upstairs);
X = lift.phi(Y);

% X is a column stochastic matrix such that X*X is (we hope) close to A.
