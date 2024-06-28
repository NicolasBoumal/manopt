clear; clc; clf;

% Compare quadprog to Hadamard lift for quadratic programs on orthant.

% See also Section 5.1 in the following paper:
% On Squared-Variable Formulations
% Lijun Ding, Stephen J. Wright
% https://arxiv.org/abs/2310.01784

% For large n, lifts beat quadprog.
n = 2000;
lift = hadamardlift('nonnegative', n);

% Random convex quadratic f(x) = .5*x'*A*x + b'*x.
% Sparsity is easily exploited in the lifts approach.
A = sprandsym(n, .01) + n*speye(n);
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
[y, f_manopt, info] = trustregions(upstairs);
x_manopt = lift.phi(y);
t_manopt = toc(t_manopt);

t_quadprog = tic();
x_quadprog = quadprog(A, b, [], [], [], [], zeros(n, 1), []);
t_quadprog = toc(t_quadprog);
f_quadprog = getCost(downstairs, x_quadprog);

fprintf('Manopt:  \n  Time: %.3e,  min(x): %.3e,  f(x): %.10e\n', ...
            t_manopt, min(x_manopt), f_manopt);
fprintf('Quadprog:\n  Time: %.3e,  min(x): %.3e,  f(x): %.10e\n', ...
            t_quadprog, min(x_quadprog), f_quadprog);

reference = min([f_quadprog, f_manopt]);
semilogy([info.time], [info.cost] - reference, '.-', ...
         t_quadprog, f_quadprog - reference, '.', ...
         'LineWidth', 2, 'MarkerSize', 15);
grid on;
xlabel('Computation time [s]');
title('Difference between cost function value and best cost reached overall.');
legend('Manopt', 'quadprog');
