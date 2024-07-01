clear; clf; clc;

% Setup random problem data.
% Unknown x_true in R^n is sparse.
% We get m < n linear measurements.
n = 100;
m = 42;
A = randn(m, n);
support_density = .1;
x_true = full(sprandn(n, 1, support_density));
b = A*x_true;

% Describe the quadratic cost function f : R^n -> R.
downstairs.M = euclideanfactory(n, 1);
downstairs.cost = @(x) .5*norm(A*x - b)^2;
downstairs.grad = @(x) A'*(A*x - b);
downstairs.hess = @(x, xdot) A'*A*xdot;

% Select the lift of R^n whose built-in regularizer is the 1-norm.
lift = hadamarddifferencelift(n);

% Call manoptlift with regularization parameter lambda > 0.
lambda = .1;
upstairs = manoptlift(downstairs, lift, [], lambda);

% Minimize g_lambda upstairs.
y = trustregions(upstairs);

% Map the answer down to a vector in R^n.
x = lift.phi(y);

% Display the results.
stem(1:n, x_true);
hold all;
stem(1:n, x, '.');
legend('True x', 'Computed x');
