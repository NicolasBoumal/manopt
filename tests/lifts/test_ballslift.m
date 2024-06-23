clear; clc; clf;

% Try to pack m points in a ball in R^n
n = 2;
m = 64;

lift = ballslift(n, m);

gram2edm = @(G) cdiag(G)*ones(1, m) + ones(m, 1)*cdiag(G).' - 2*G;
sigma = .05;
downstairs.cost = @(X) log(sum(exp(-gram2edm(X.'*X)/(2*sigma^2)), 'all'));

upstairs = manoptlift(downstairs, lift, 'AD');

Y = trustregions(upstairs);

X = lift.phi(Y);

if n == 2
    plot(X(1, :), X(2, :), '.', 'MarkerSize', 20);
    t = linspace(0, 2*pi, 251);
    hold all;
    plot(cos(t), sin(t), 'k-', 'LineWidth', 2);
    plot(0, 0, 'k.', 'MarkerSize', 10);
    axis equal off;
    set(gcf, 'Color', 'w');
end
