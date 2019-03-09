n = 3;
M = rotationsfactory(n, 1);
A = randn(n);
problem.cost = @(X) .5*norm(X-A, 'fro')^2 + trace(X) + cos(X(1, 1));
Z = zeros(n); Z(1, 1) = 1;
problem.egrad = @(X) X-A + eye(n) - sin(X(1, 1))*Z;
at = @(M, i, j) M(i, j);
problem.ehess = @(X, Sdot) X*Sdot - cos(X(1, 1))*at(X*Sdot, 1, 1)*Z;


M.exp = M.retr_qr; % check*** tools use the exponential if available!
problem.M = M;
X = M.rand();
V = M.randvec(X);
% checkgradient(problem); pause;
checkhessian(problem, X, V);
problem.M.norm(X, getGradient(problem, X))

% Numerical check (finite differences) of acceleration of retraction at X
% along V. For polar retraction, we expect ~0 acceleration since its second
% order. For QR retraction, I think it's only first-order, and indeed we
% get non-zero acceleration, BUT the above checkhessian test succeeds???
%%
M = rotationsfactory(n, 1); % make sure we work with a clean manifold
M.retr = M.retr_polar;
acc = @(t) M.norm(X, M.proj(X, (M.retr(X, V, t) - 2*X + M.retr(X, V, -t))/t^2));
t = logspace(-16, 0, 51);
g = zeros(size(t));
for k = 1 : numel(t)
    g(k) = acc(t(k));
end
loglog(t, g);
xlabel('t'); title('Finite difference acceleration norm at 0');
