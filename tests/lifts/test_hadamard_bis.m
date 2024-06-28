clear; clc; clf;

% Compare quadprog to Hadamard lift for quadratic programs on orthant.

% See also Section 5.1 in the following paper:
% On Squared-Variable Formulations
% Lijun Ding, Stephen J. Wright
% https://arxiv.org/abs/2310.01784

% For large n, lifts beat quadprog.
n = 10000;
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

t_manopt_tr = tic();
[y_tr, f_manopt_tr, info_tr] = trustregions(upstairs);
x_manopt_tr = lift.phi(y_tr);
t_manopt_tr = toc(t_manopt_tr);

t_manopt_cg = tic();
[y_cg, f_manopt_cg, info_cg] = conjugategradient(upstairs);
x_manopt_cg = lift.phi(y_cg);
t_manopt_cg = toc(t_manopt_cg);

t_quadprog = tic();
x_quadprog = quadprog(full(A), b, [], [], [], [], zeros(n, 1), []);
t_quadprog = toc(t_quadprog);
f_quadprog = getCost(downstairs, x_quadprog);

fprintf('Manopt TR:\n  Time: %.3e,  min(x): %.3e,  f(x): %.10e\n', ...
            t_manopt_tr, min(x_manopt_tr), f_manopt_tr);
fprintf('Manopt CG:\n  Time: %.3e,  min(x): %.3e,  f(x): %.10e\n', ...
            t_manopt_cg, min(x_manopt_cg), f_manopt_cg);
fprintf('Quadprog: \n  Time: %.3e,  min(x): %.3e,  f(x): %.10e\n', ...
            t_quadprog, min(x_quadprog), f_quadprog);

reference = min([f_quadprog, f_manopt_tr, f_manopt_cg]);
semilogy([info_tr.time], [info_tr.cost] - reference, '.-', ...
         [info_cg.time], [info_cg.cost] - reference, '.-', ...
         t_quadprog, f_quadprog - reference, '.', ...
         'LineWidth', 2, 'MarkerSize', 25);
grid on;
xlabel('Computation time [s]');
title('Difference between cost function value and best cost reached overall.');
legend('Manopt TR', 'Manopt CG', 'quadprog');
