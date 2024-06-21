clear; clc; clf;

% Try to pack m points in a ball in R^n
n = 2;
m = 50;

lift = ballslift(n, m);

gram2edm = @(G) diag(G)*ones(1, m) + ones(m, 1)*diag(G)' - 2*G;
downstairs.cost = @(X) -min(gram2edm(X.'*X) + 4*eye(m), [], 'all');
% Would be good to smooth the cost function and provide grad (or use AD).

upstairs = manoptlift(downstairs, lift);

Y = rlbfgs(upstairs);

X = lift.phi(Y);

if n == 2
    plot(X(1, :), X(2, :), '.', 'MarkerSize', 20);
    t = linspace(0, 2*pi, 251);
    hold all;
    plot(cos(t), sin(t), 'k-', 'LineWidth', 2);
    axis equal;
end
