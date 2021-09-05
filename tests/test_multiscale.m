% Real case
n = 3;
m = 10;
A = randn(n, n, m);
b = randn(m, 1);
B = multiscale(b, A);
C = zeros(n, n, m);
for k = 1 : m
    C(:, :, k) = A(:, :, k) * b(k);
end
D = B - C;
norm(D(:))

% Complex case
n = 3;
m = 10;
A = randn(n, n, m) + 1i*randn(n, n, m);
b = randn(m, 1) + 1i*randn(m, 1);
B = multiscale(b, A);
C = zeros(n, n, m);
for k = 1 : m
    C(:, :, k) = A(:, :, k) * b(k);
end
D = B - C;
norm(D(:))
