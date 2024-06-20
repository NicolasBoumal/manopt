function test_replacesparseentries()

m = 10000;
n = 10000;
M = rand(m, n);
M(M > .01) = 0;
M = sparse(M);

k = nnz(M);

x = randn(k, 1);

profile clear;
profile on;

for iter = 1:500
    L = replacesparseentries(M, x);
end

for iter = 1:500
    [i, j] = find(M);
    K = sparse(i, j, x, m, n, k);
end

for iter = 1:500
    replacesparseentries(M, x, 'inplace');
end

profile off;
profile report;


norm(L - K, 'fro')

end
