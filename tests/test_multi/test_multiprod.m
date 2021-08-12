n = 50;
m = 65;
p = 105;
k = 10;
X = randn(n, m, k);
Y = randn(m, p, k);

% On laptop Aug. 12, 2021:
% n = 50;
% m = 65;
% p = 105;
% k = 10;
% Elapsed time is   8.98 seconds. % new multiprod
% Elapsed time is 678.95 seconds. % old multiprod
% Elapsed time is  41.36 seconds. % for-loop multiprod
% Elapsed time is  55.37 seconds. % mmx multiprod
% Relative difference: 8.78498e-17
% Relative difference: 2.04793e-16
% Relative difference: 2.04793e-16

t = tic();
for K = 1 : 100000
    XY1 = multiprod(X, Y);
end
toc(t)

t = tic();
for K = 1 : 100000
    XY4 = multiprod_legacy(X, Y);
end
toc(t)

t = tic();
for K = 1 : 100000
    XY2 = multiprod_ez(X, Y);
end
toc(t)

t = tic();
for K = 1 : 100000
    XY3 = mmx('mult', X, Y);
end
toc(t)

fprintf('Relative difference: %g\n', norm(XY1(:) - XY2(:))/norm(XY2(:)));
fprintf('Relative difference: %g\n', norm(XY3(:) - XY2(:))/norm(XY2(:)));
fprintf('Relative difference: %g\n', norm(XY4(:) - XY2(:))/norm(XY2(:)));

% pagemtimes is faster

% mmx is sometimes slower than the ez version ...
