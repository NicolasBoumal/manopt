clear; clf; clc;

n = 1000;

A = randsym(n);
A = A*A';
b = randn(n, 1);

downstairs.M = euclideanfactory(n, 1);
downstairs.cost = @(x) .5*x'*A*x + b'*x;
downstairs.grad = @(x) A*x + b;
downstairs.hess = @(x, xdot) A*xdot;

lift = hadamarddifferencelift(n);

% This should be created by the lift itself, so that it would be consistant
% with the lift's domain and the embedded flag, without separating them.
lambda = 1.1354;
lift.rho = @(y) lambda*norm(y, 'fro')^2;
lift.gradrho = @(y) 2*lambda*y;
lift.hessrho = @(y, ydot) 2*lambda*ydot;

upstairs = manoptlift_with_regularizer(downstairs, lift, 'noAD');

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