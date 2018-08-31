%% test thread management
max_thread=64;
for i=1:max_thread
    mmx(i);
    A = randn(1);
    B = randn(1);
    assert(mmx('mult',A,B) == A*B)
     assert(mmx('b',A,B) == A\B)
     assert(mmx('s',A,[]) == A*A')
     mmx((max_thread+1)-i);
     assert(mmx('mult',A,B) == A*B)
     assert(mmx('b',A,B) == A\B)
     assert(mmx('s',A,[]) == A*A')
end

%% test thread management more intensely
error = 1e-6;
for i=1:max_thread
    mmx(round(rand*max_thread));
%     pause(.1)
%      mmx(round(rand*max_thread));
    A = randn(100,100);
    B = randn(100,50);
    diff = norm(mmx('mult',A,B) - A*B);
    assert(abs(diff(:)) < error)
    diff = norm(mmx('b',A,B) - A\B);
    assert(abs(diff(:)) < error)
    diff = norm(mmx('s',A,[]) - A*A');
    assert(abs(diff(:)) < error)

    mmx((max_thread+1)-i);
    diff = norm(mmx('mult',A,B) - A*B);
    assert(abs(diff(:)) < error)
    diff = norm(mmx('b',A,B) - A\B);
    assert(abs(diff(:)) < error)
    diff = norm(mmx('s',A,[]) - A*A');
    assert(abs(diff(:)) < error)
end

%% test matrix multiplication

assert(mmx('mult',2,3) == 6)

% with singleton expansion:

A = randn(5,4,3,10,1);
B = randn(4,6,3,1 ,6);
C = zeros(5,6,3,10,6);

for i = 1:3
   for j = 1:10
      for k = 1:6
         C(:,:,i,j,k) = A(:,:,i,j,1) * B(:,:,i,1,k);
      end
   end
end

diff = C - mmx('mult',A,B);

fprintf('difference is %f\n',norm(diff(:)))


%% test matrix squaring

A = randn(4,5);
B = randn(4,5);

diff = norm(mmx('square',A,[]) - A*A');
fprintf('difference is %f\n',norm(diff(:)))

diff = norm(mmx('square',A,B) - 0.5*(A*B'+B*A'));
fprintf('difference is %f\n',norm(diff(:)))

diff = norm(mmx('square',A,[],'t') - A'*A);
fprintf('difference is %f\n',norm(diff(:)))

diff = norm(mmx('square',A,B,'t') - 0.5*(A'*B+B'*A));
fprintf('difference is %f\n',norm(diff(:)))


%% test cholesky factor

A = randn(4,4);
A = A*A';
diff = norm(mmx('chol',A,[]) - chol(A));
fprintf('difference is %f\n',norm(diff(:)))

%% test backslash 

fprintf('triangular: ')
A = randn(4,4);
A = triu(A);
B = randn(4,2);

diff = norm(mmx('backslash',A,B,'u') - A\B);
fprintf('difference is %f\n\n',norm(diff(:)))

fprintf('symmetric positive definite: ')
A = randn(4,4);
A = A*A';
B = randn(4,2);

diff = norm(mmx('backslash',A,B,'p') - A\B);
fprintf('difference is %f\n\n',norm(diff(:)))

fprintf('square: ')
A = randn(4,4);
B = randn(4,2);

diff = norm(mmx('backslash',A,B) - A\B);
fprintf('difference is %f\n\n',norm(diff(:)))

fprintf('overdetermined: ')
A = randn(4,3);
B = randn(4,2);

diff = norm(mmx('backslash',A,B) - A\B);
fprintf('difference is %f\n\n',norm(diff(:)))

fprintf('underdetermined: ')
A = randn(4,6);
B = randn(4,2);

diff = norm(mmx('backslash',A,B) - pinv(A)*B);
fprintf('difference is %f\n\n',norm(diff(:)))

