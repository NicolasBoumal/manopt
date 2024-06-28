clear; clc; clf;

% Compare quadprog to Hadamard lift for quadratic programs on orthant.

% For large n, say, 5'000, lifts beat quadprog.
n = 5000;
lift = hadamardlift('nonnegative', n);

% random stochastic matrix
A = randsym(n) + n*eye(n);
b = randn(n, 1);

downstairs.costgrad = @(x) costgrad(A, b, x);
function [f, g] = costgrad(A, b, x)
    Ax = A*x;
    f = x'*(.5*Ax + b);
    if nargout == 2
        g = Ax + b;
    end
end
downstairs.hess = @(x, xdot) A*xdot;

upstairs = manoptlift(downstairs, lift);

t_manopt = tic();
y = trustregions(upstairs);
x_manopt = lift.phi(y);
t_manopt = toc(t_manopt);

t_quadprog = tic();
x_quadprog = quadprog(A, b, [], [], [], [], zeros(n, 1), []);
t_quadprog = toc(t_quadprog);

fprintf('Manopt:\n  Time: %.3e,  min(x): %.3e,  f(x): %.3e\n', ...
            t_manopt, min(x_manopt), getCost(downstairs, x_manopt));
fprintf('Quadprog:\n  Time: %.3e,  min(x): %.3e,  f(x): %.3e\n', ...
            t_quadprog, min(x_quadprog), getCost(downstairs, x_quadprog));
