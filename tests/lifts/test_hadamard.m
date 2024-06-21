clear; clc; clf;

n = 4;
m = 2;

lift = hadamardlift('simplex', n, m);
A = randn(n, m);
inner = @(U, V) U(:).'*V(:);
sqfrobnorm = @(U) inner(U, U);
downstairs.cost = @(X) sqfrobnorm(X-A);
[upstairs, downstairs] = manoptlift(downstairs, lift, 'AD');

X = trustregions(upstairs);

A
Y = lift.phi(X)
