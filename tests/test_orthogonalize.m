% Test how accurately we can orthogonalize ill-conditioned bases of tangent
% vectors. We test three metrics:
%  1) Is Q actually orthonormal (the figure displays log(|Q'Q|); since Q'Q
%     should be identity, we hope to see 0 on diagonal and -inf everywhere
%     else.)
%  2) Is A = QR? This is in the title of the plots.
%  3) How much time does it take to compute?
%
% Nicolas Boumal, Oct. 5, 2017

clear all;
clc;

M = spherefactory(100);

% M = stiefelcomplexfactory(200, 56);
% M = productmanifold(struct('S', spherefactory(5), 'R', rotationsfactory(3, 10)));

x = M.rand();

% Create a poorly conditioned basis
A = cell(M.dim()-5, 1);
A{1} = M.randvec(x);
for k = 2 : numel(A)
    A{k} = M.lincomb(x, 1, A{k-1}, 1e-6, M.randvec(x));
end

t1 = tic();
[Q1, R1] = orthogonalize(M, x, A);
t1 = toc(t1);

t2 = tic();
[Q2, R2] = orthogonalizetwice(M, x, A);
t2 = toc(t2);

t3 = tic();
[Q3, R3] = orthogonalize_legacy(M, x, A);
t3 = toc(t3);

% To check if Q is really orthonormal
G1 = grammatrix(M, x, Q1);
G2 = grammatrix(M, x, Q2);
G3 = grammatrix(M, x, Q3);

% To check of QR = A
distsq = 0;
for k = 1 : numel(A)
    distsq = distsq + M.norm(x, M.lincomb(x, 1, lincomb(M, x, Q1(1:k), R1(1:k, k)), -1, A{k}))^2;
end
dist1 = sqrt(distsq);

distsq = 0;
for k = 1 : numel(A)
    distsq = distsq + M.norm(x, M.lincomb(x, 1, lincomb(M, x, Q2(1:k), R2(1:k, k)), -1, A{k}))^2;
end
dist2 = sqrt(distsq);

distsq = 0;
for k = 1 : numel(A)
    distsq = distsq + M.norm(x, M.lincomb(x, 1, lincomb(M, x, Q3(1:k), R3(1:k, k)), -1, A{k}))^2;
end
dist3 = sqrt(distsq);


subplot(1, 3, 1);
imagesc(log10(abs(G1))); axis equal, axis tight; colorbar; set(gca, 'CLim', [-20, 0]);
title(sprintf('MGS: CPU time: %g\n||A - QR|| = %g', t1, dist1));
subplot(1, 3, 2);
imagesc(log10(abs(G2))); axis equal, axis tight; colorbar; set(gca, 'CLim', [-20, 0]);
title(sprintf('MGS-twice: CPU time: %g\n||A - QR|| = %g', t2, dist2));
subplot(1, 3, 3);
imagesc(log10(abs(G3))); axis equal, axis tight; colorbar; set(gca, 'CLim', [-20, 0]);
title(sprintf('Legacy: CPU time: %g\n||A - QR|| = %g', t3, dist3));
