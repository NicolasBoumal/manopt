clear; clc;

m = 1000;
n = 1500;
r = 5;

A = sprand(m, n, .01); % random sparse matrix

Rmn = euclideanlargefactory(m, n); % mind the 'large'


dwnstrs.M = Rmn;
dwnstrs.cost = @(X) .5*Rmn.dist(X, A)^2;
dwnstrs.grad = @(X) Rmn.diff(X, A);
dwnstrs.hess = @(X, Xdot) Xdot;

lift = burermonteiroLRlift(m, n, r);
upstairs = manoptlift(dwnstrs, lift);

% initialization close to zero
LR0.L = randn(m, r)/sqrt(m*r)/1000;
LR0.R = randn(n, r)/sqrt(n*r)/1000;

LR = trustregions(upstairs, LR0);

% Confirm that the positive singular values of L*R'
% match the r top singular values of A.
% For efficiency, we do not form X = LR.L * LR.R'.
% Instead, we apply a QR to both L and R, and
% deduce the singular values of X from there.
[Ql, Rl] = qr(LR.L, 0);
[Qr, Rr] = qr(LR.R, 0);
svd(Rl*Rr')'
svds(A, r+1)'