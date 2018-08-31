n = 10;
m = 20;
k = 1000;
X = randn(n, m, k);

t = tic();
Xt1 = multitransp(X);
toc(t)

t = tic();
Xt2 = multitransp_ez(X);
toc(t)

fprintf('Relative difference: %g\n', norm(Xt1(:) - Xt2(:))/norm(Xt2(:)));

% mutlitransp is better here (it's doing the right thing, calling permute)