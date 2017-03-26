function Test_linearFunction

% Define a linear space
m = 5;
n = 2;
manifold = euclideanfactory(m,n);

% random symmetric square matrix
A = randn(m,n);

% Create the problem structure.
problem.M = manifold;

% Define the problem cost function, Euclidean gradient and Hessian.
problem.cost = @cost;
function f = cost(X)
  f = trace(A'*X);
end

problem.egrad = @egrad;
function G = egrad(X)
  G = A;
end

problem.ehess = @ehess;
function H = ehess(X, Xdot)
  H = zeros(size(Xdot));
end

% Check gradient and Hessian correctness
% This is a quadratic function in a linear manifold,
% so the quadratic approximation used in `checkhessian` should be exact
% Other similar cases are those where the unknowns belong to a tangent
% space (also a linear manifold).
checkgradient( problem ); pause
checkhessian(  problem ); pause

end