% Oct. 22, 2023
clear all; close all; clc;

M = rotationsfactory(3);
PT = @(X, Y, V) (Y'*X)*expm(M.log(X, Y)/2)*V*expm(M.log(X, Y)/2);

t = 1e-4;
X = M.rand();
q = M.randvec(X);
Y = M.exp(X, q, t);
q = M.randvec(Y);
Z = M.exp(Y, q, t);
V = M.randvec(X);
W = PT(Z, X, PT(Y, Z, PT(X, Y, V)));
M.norm(X, V - W) / t^2
