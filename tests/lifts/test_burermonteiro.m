clear; clf; clc;

n = 1000;
p = 15;

C = abs(sprandsym(n, .001));

Rnn = euclideanlargefactory(n, n);
downstairs.M = Rnn;
downstairs.cost = @(X) Rnn.inner(C, X);
downstairs.grad = @(X) C;
downstairs.hess = @(X, Xdot) Rnn.zero();

lift = burermonteirolift('unitdiag', n, p, 'symmetric');

upstairs = manoptlift(downstairs, lift);

Y0 = upstairs.M.rand();

manual.M = upstairs.M;
inner = @(U, V) U(:).'*V(:); % sum(U.*V, 'all');
manual.cost = @(Y) inner(C*Y, Y);
manual.egrad = @(Y) 2*C*Y;
manual.ehess = @(Y, Ydot) 2*C*Ydot;

% profile clear;
% profile on;
[~, ~, infomanual] = trustregions(manual, Y0);
[Y, ~, info] = trustregions(upstairs, Y0);
% profile off;
% profile report;

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
