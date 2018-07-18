clear all; close all; clc;

% July 8, 2018, NB
n = 5;
p = 3;
k = 2;
M = rotationsfactory(n, k); %, true);

% M.retr = M.retr_qr;
% M.invretr = M.invretr_qr;

M.retr = M.retr_polar;
M.invretr = M.invretr_polar;

X = M.rand();
S = M.randvec(X);
S = M.lincomb(X, randn(1), S);
Y = M.retr(X, S);
V = M.invretr(X, Y);
fprintf('Retraction inversion error: %g\n', ...
        M.norm(X, M.lincomb(X, 1, S, -1, V)));
