%% MMX - Multithreaded matrix operations on N-D matrices
% mmx treats an N-D matrix of double precision values as a set of pages 
% of 2D matrices, and performs various matrix operations on those pages.  
% mmx uses multithreading over the higher dimensions to achieve good
% performance. Full singleton expansion is available for most operations.

%% Fast N-D Multiplication
n  = 80;                % rows
m  = 40;                % columns
N  = 10000;             % pages
A  = randn(n,m,N);
B  = randn(m,n,N);
tic;
C  = mmx('mult', A, B);
toc
%%
C2    = zeros(n,n,N);
tic;
for i=1:N
   C2(:,:,i) = A(:,:,i)*B(:,:,i);
end
toc    
%%
dispx = @(x) fprintf('difference = %g\n',x);
dispx(max(abs(C(:)-C2(:))))

%% Multi-threading along the pages
% Other packages like Peter Boettcher's venerable <http://www.mit.edu/~pwb/matlab/ ndfun> 
% Or James Tursa's
% <http://www.mathworks.com/matlabcentral/fileexchange/25977 mtimesx> rely
% on multithreading *inside the threads* using multithreaded BLAS
% libraries. It turns out that if you want to operate on many small
% matrices, it makes more sense to let each thread operate on a matrix
% independently. Actually it's possible
% <http://www.mathworks.com/matlabcentral/fileexchange/25977 mtimesx> tries
% to do this using OMP but it doesn't seem to work that well.
tic;
mtimesx(A, B, 'speedomp');
toc

%% Full performance comparison
compare_mult_flops;

%% 
% You can see how around dimension 35, when the low-level multi-threading
% kicks in, the CPU get flooded with threads and efficiency drops.

%% Singleton Expansion
% Singleton expansion is supported for |dimensions > 2|

A = randn(5,4,3,10,1);
B = randn(4,6,1,1 ,6);
C = zeros(5,6,3,10,6);

for i = 1:3
   for j = 1:10
      for k = 1:6
         C(:,:,i,j,k) = A(:,:,i,j,1) * B(:,:,1,1,k);
      end
   end
end

diff = C - mmx('mult',A,B);

dispx(norm(diff(:)))


%% Transpose Flags
% |C = MMX('mult', A, B, mod)| where mod is a modifier string, will
% transpose one or both of A and B. Possible values for mod are
% 'tn', 'nt' and  'tt' where 't' stands for *transposed* and 'n' for
% *not-transposed* . For example 
A = randn(n,n);
B = randn(n,n);
dispx(norm(mmx('mult',A,B)      - A *B));
dispx(norm(mmx('mult',A,B,'tn') - A'*B));
dispx(norm(mmx('mult',A,B,'tt') - A'*B'));
dispx(norm(mmx('mult',A,B,'nt') - A *B'));


%% Matrix Squaring
A = randn(n,m);
B = randn(n,m);
dispx(norm(mmx('square',A,[])     - A*A'            ));
dispx(norm(mmx('square',A, B)     - 0.5*(A*B'+B*A') ));
dispx(norm(mmx('square',A,[],'t') - A'*A            ));
dispx(norm(mmx('square',A, B,'t') - 0.5*(A'*B+B'*A) ));
%%
% Results do not always equal Matlab's results, but are within machine
% precision thereof.


%% Cholesky factorization
A = randn(n,n);
A = A*A';
dispx(norm(mmx('chol',A,[]) - chol(A)));

%%
% Timing comparison:
A  = randn(n,n,N);
A  = mmx('square',A,[]);
tic;
C  = mmx('chol',A,[]);
toc
C2    = zeros(n,n,N);
tic;
for i=1:N
   C2(:,:,i) = chol(A(:,:,i));
end
toc

%% Backslash 
% Unlike other commands, 'backslash' does not support singleton
% expansion. If A is square, mmx will use LU factorization, otherwise it
% will use QR factorization. 
B = randn(n,m);
A = randn(n,n);
%%
% General:
dispx(norm(mmx('backslash',A,B) - A\B));
%%
% Triangular:

% upper:
Au = triu(A) + abs(diag(diag(A))) + eye(n); %no small values on the diagonal
dispx(norm(mmx('backslash',Au,B,'u') - Au\B));
% lower:
Al = tril(A) + abs(diag(diag(A))) + eye(n); %no small values on the diagonal
dispx(norm(mmx('backslash',Al,B,'l') - Al\B));
%%
% Symmetric Positive Definite:
AA = A*A';
dispx(norm(mmx('backslash',AA,B,'p') - AA\B));
%%
% Cholesky/LU timing comparison:
A  = randn(n,n,N);
A  = mmx('square',A,[]);
B  = randn(n,1,N);
tic;
mmx('backslash',A,B); % uses LU
toc
tic;
mmx('backslash',A,B,'p'); % uses Cholesky
toc

%%
% Overdetermined:
A = randn(n,m);
B = randn(n,m);

dispx(norm(mmx('backslash',A,B) - A\B));

%%
% Underdetermined:
A = randn(m,n);
B = randn(m,n);

dispx(norm(mmx('backslash',A,B) - pinv(A)*B));
%%
% In the underdetermined case, (i.e. when
% |size(A,1) < size(A,2))|, mmx will give the least-norm solution which
% is equivalent to |C = pinv(A)*B|, unlike matlab's mldivide. 

%%% Thread control
% mmx will automatically start a number of
% threads equal to the number of available processors, however the
% number can be set manually to n using the command |mmx(n)|.
% The command |mmx(0)| clears the threads from memory. Changing the
% threadcount quickly without computing anything, as in
%%
% 
%  for i=1:5
%     mmx(i);
%  end
%%
% can cause problems. Don't do it.

%% Checking of special properties
% The functions which assume special types of square
% matrices as input ('chol' and 'backslash' for 'U','L' or 'P'
% modifiers) do not check that the inputs are indeed what you say they
% are, and produce no error if they are not. Caveat computator.

%% Compilation
% To compile run 'build_mmx'. Type 'help build_mmx' to read
% about compilation issues and options

%% Rant
% Clearly there should be someone at Mathworks whose job it is to do this
% stuff. As someone who loves Matlab deeply, I hate to see its foundations
% left to rot. Please guys, allocate engineer-hours to the Matlab core, rather than the
% toolbox fiefdoms. We need full singleton expansion everywhere. Why isn't
% it the case that
%%
% 
%  [1 2] + [0 1]' == [1 2;2 3] ?
%
% bsxfun() is a total hack, and polluting
% everybody's code. We need expansion on the pages like mmx(), but
% with transparent and smart use of *both* CPU and GPU. GPUArray? Are you
% kidding me? I shouldn't have to mess with that. Why is it that (for years
% now), the fastest implementation of repmat(), has been Minka's
% <http://research.microsoft.com/en-us/um/people/minka/software/lightspeed/
% Lightspeed toolbox>? Get your act together soon guys, or face 
% obsolescence.