clear; clc;

n = 1000;

A = randsym(n);
A = A*A';
b = randn(n, 1);

downstairs.M = euclideanfactory(n, 1);
downstairs.cost = @(x) .5*x'*A*x + b'*x;
downstairs.grad = @(x) A*x + b;
downstairs.hess = @(x, xdot) A*xdot;

lift = hadamarddifferencelift(n);

lambda = 1.1354;

upstairs = manoptlift(downstairs, lift, 'noAD', lambda);

y = trustregions(upstairs);

x_manopt = lift.phi(y);


cvx_begin
    variable x(n);
    minimize( .5*x'*A*x + b'*x + lambda*norm(x, 1) );
cvx_end

x_cvx = x;

fprintf('\n\n');

fprintf('Relative difference between two solutions: %.3e\n', ...
        norm(x_manopt - x_cvx)/norm(x_cvx));

f = @(x) .5*x'*A*x + b'*x + lambda*norm(x, 1);

fprintf('Manopt cost: %.12g\n', f(x_manopt));
fprintf('   CVX cost: %.12g\n', f(x_cvx));