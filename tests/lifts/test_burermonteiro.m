clear; clf; clc;

n = 10000;
p = 15;

constraint = 'unitdiag';
lift = burermonteirolift(constraint, n, p, 'symmetric');
Rnn = lift.N;
downstairs.M = Rnn;

% sign = +1;
% sqfrobnorm = @(Z) Z(:)'*Z(:);
% downstairs.cost = @(X) sign*.5*Rnn.norm(X)^2;
% downstairs.grad = @(X) Rnn.scale(sign, X);
% downstairs.hess = @(X, Xdot) Rnn.scale(sign, Xdot);

C = sparse(double(randsym(n) < -2));
C(1:(n+1):end) = 0;
downstairs.cost = @(X) Rnn.inner(C, X);
downstairs.grad = @(X) C;
downstairs.hess = @(X, Xdot) Rnn.zero();

[upstairs, downstairs] = manoptlift(downstairs, lift);

% checkgradient(downstairs);
% checkhessian(downstairs);
% checkgradient(upstairs);
% checkhessian(upstairs);

Y0 = upstairs.M.rand();

manual.M = upstairs.M;
inner = @(U, V) U(:).'*V(:); % sum(U.*V, 'all');
manual.cost = @(Y) inner(C*Y, Y);
manual.egrad = @(Y) 2*C*Y;
manual.ehess = @(Y, Ydot) 2*C*Ydot;

profile clear; pause;
profile on;
[~, ~, infomanual] = trustregions(manual, Y0);
[Y, ~, info] = trustregions(upstairs, Y0);
profile off;
profile report;

semilogy([info.time], [info.gradnorm], '.-', ...
         [infomanual.time], [infomanual.gradnorm], '.-');
legend('lift', 'manual');
grid on;

if p == 2
    scatter(Y(:, 1), Y(:, 2));
    hold all;
    t = linspace(0, 2*pi, 251);
    plot(cos(t), sin(t), 'k-');
    plot(0, 0, 'k.', 'MarkerSize', 10);
    axis equal;
    axis off;
    box off;
    set(gcf, 'Color', 'w');
end
