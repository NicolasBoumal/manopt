clear; close all; clc;

n = 500;
complex = true;

A = randn(n);
X0 = randn(n);
if complex
    A = A + 1i*randn(n);
    X0 = X0 + 1i*randn(n);
end
A = A+A';
C = A*X0+X0*A;

X1 = lyapunov_symmetric(A, C);

norm(A*X1+X1*A-C, 'fro') / norm(C, 'fro')
norm(X0, 'fro')
norm(X1, 'fro')