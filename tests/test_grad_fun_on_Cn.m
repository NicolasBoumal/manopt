clear; close all; clc;
n = 10;
m = 120;
A = randn(n, n, m) + 1i*randn(n, n, m);
b = randn(m, 1) + 1i*randn(m, 1);
problem.M = euclideancomplexfactory(n, 1);
z = @(x) squeeze(multiprod(x', multiprod(A, x))) - b;
problem.cost = @(x) real(z(x)'*z(x));
% problem.grad = ...
%    @(x) 2*squeeze(multiprod(A, x))*conj(z(x)) + ...
%         2*squeeze(multiprod(multihconj(A), x))*z(x);
% checkgradient(problem);

% check if AD works on this example in R2021b.
problem = preprocessAD(problem);
checkgradient(problem);

%{
% Equivalent code without anything fancy
pause;
problem.cost = @(x) cost(x, A, b);
problem.grad = @(x) grad(x, A, b);
checkgradient(problem);
function f = cost(x, A, b)
    f = 0;
    m = size(A, 3);
    z = zeros(m, 1);
    for k = 1 : m
        z(k) = x'*A(:, :, k)*x - b(k);
        f = f + abs(z(k))^2;
    end
end
function g = grad(x, A, b)
    g = zeros(size(x));
    m = size(A, 3);
    z = zeros(m, 1);
    for k = 1 : m
        z(k) = x'*A(:, :, k)*x - b(k);
        g = g + 2*( conj(z(k))*A(:, :, k)*x + z(k)*A(:, :, k)'*x);
    end
end
%}

% Special case:
% A = multiherm(A);
% b = real(b);
% problem.grad = @(x) 4*squeeze(multiprod(A, x))*z(x);