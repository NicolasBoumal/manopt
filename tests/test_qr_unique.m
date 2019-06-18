clear all;
close all;
clc;
% NB, June 18, 2019.

fprintf('If all is well with qr_unique, this script\nought to print many (numerical) zeros.\n');

mm = [5 8 5 5 8 5];
nn = [5 5 8 5 5 8];
NN = [1 1 1 9 9 9];
for k = 1 : length(mm)
    m = mm(k);
    n = nn(k);
    N = NN(k);
    A = randn(m, n, N) + 1i*randn(m, n, N); % also test corner cases, with some of these randn -> zeros
    [Q, R] = qr_unique(A);
    
    for j = 1 : N
        q = Q(:, :, j);
        r = R(:, :, j);
        
        [qq, rr] = qr(A(:, :, j), 0);
        assert(all(size(qq) == size(q)), 'Q size mismatch.');
        assert(all(size(rr) == size(r)), 'R size mismatch.');
        
        disp(norm(q*r - qq*rr));
        disp(norm(q'*q - eye(size(q, 2))));
        disp(norm(min(0, real(diag(r)))));
        disp(norm(imag(diag(r))));
        
    end
    
end
