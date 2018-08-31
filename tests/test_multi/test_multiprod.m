n = 3;
m = 3;
p = 3;
k = 2;
X = randn(n, m, k);
Y = randn(m, p, k);

t = tic();
for K = 1 : 100000
    XY1 = multiprod(X, Y);
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

% seems like the ez version is faster, but it's not completely clear cut
% ......

% mmx is sometimes slower than the ez version ...