clear all; close all; clc;

% Online mean computation
% July 17, 2018, NB
n = 5;
p = 3;
k = 2;
M = stiefelfactory(n, p, k);

% M.retr = M.retr_qr;
% M.invretr = M.invretr_qr;

M.retr = M.retr_polar;
M.invretr = M.invretr_polar;

sigma = .1;
X0 = M.rand(); % true center
S = M.randvec(X0);
Y = M.retr(X0, S, sigma*randn(1)); % this is how new samples are generated
X = Y; % at first, our estimate X of X0 is just the first sample
g = M.norm(X0, M.invretr(X0, X));
for k = 2 : 100000
    S = M.randvec(X0);
    Y = M.retr(X0, S, sigma*randn(1)); % new sample
    X = M.retr(X, M.invretr(X, Y), 1/k); % heuristic online averaging formula
    g(end+1) = M.norm(X0, M.invretr(X0, X)); % track a kind of distance to the true mean
end

loglog(1:numel(g), g, '-');

P = polyfit(log(1:numel(g)), log(g), 1);
title(sprintf('Average loglog slope: %g (hope to see -1/2)', P(1)));