function fake_output = mmx(varargin)
%MMX - Multithreaded matrix operations on N-D matrices
%    MMX treats an N-D matrix of double precision values as a set of pages 
%    of 2D matrices, and performs various matrix operations on those pages.  
%    MMX uses multithreading over the higher dimensions to achieve good
%    performance. Full singleton expansion is available for most operations. 
% 
%    C = MMX('mult', A, B) is equivalent to the matlab loop
%    for i=1:N,
%        C(:,:,i) = A(:,:,i) * B(:,:,i);
%    end
%    Singleton expansion is enabled on all dimensions so for example if
%    A = randn(5,4,3,10,1);
%    B = randn(4,6,3,1 ,6);
%    C = zeros(5,6,3,10,6);
%    then C = mmx('mult',A,B) equivalent to 
%    for i = 1:3
%       for j = 1:10
%          for k = 1:6
%             C(:,:,i,j,k) = A(:,:,i,j,1) * B(:,:,i,1,k);
%          end
%       end
%    end
% 
%    C = MMX('mult', A, B, mod) and where mod is a modifier string, will
%    transpose one or both of A and B. Possible values for mod are
%    'tn', 'nt' and  'tt' where 't' stands for 'transposed' and 'n' for
%    'not-transposed'. For example 
%    >> size(mmx('mult',randn(4,2),randn(4,2),'tn'))
%    ans =   2     2
%
%    C = MMX('square', A, [])     will perform C = A*A'
%    C = MMX('square', A, [],'t') will perform C = A'*A
%
%    C = MMX('square', A, B)       will perform C = 0.5*(A*B'+B*A')
%    C = MMX('square', A, B, 't')  will perform C = 0.5*(A'*B+B'*A)
%
%    C = MMX('chol',   A, []) will perform C = chol(A)
%
%    C = MMX('backslash', A, B) will perform C = A\B
%    Unlike other commands, 'backslash' does not support singleton
%    expansion. If A is square, mmx will use LU factorization, otherwise it
%    will use QR factorization. In the underdetermined case, (i.e. when
%    size(A,1) < size(A,2)), mmx will give the least-norm solution which
%    is equivalent to C = pinv(A)*B, unlike matlab's mldivide. 
%
%    C = MMX('backslash', A, B, 'U') or MMX('backslash', A, B, 'L') will
%    perform C = A\B assuming that A is upper or lower triangular,
%    respectively.
%    
%    C = MMX('backslash', A, B, 'P') will perform C = A\B assuming that A
%    is symmetric-positive-definite.
%
%    MMX(n) does thread control: mmx will automatically start a number of
%    threads equal to the number of available processors, however the
%    number can be set manually to n using the command mmx(n). mmx(0) will
%    clear the threads from memory.
%
%    IMPORTANT NOTE: The functions which assume special types of square
%    matrices as input ('chol' and 'backslash' for 'U','L' or 'P'
%    modifiers) do not check that the inputs are indeed what you say they
%    are, and produce no error if they are not. Caveat computator.
%
%    COMPILATION: To compile run 'build_mmx'. Type 'help build_mmx' to read
%    about compilation issues and options

error(sprintf('MEX file not found.\nTry ''build_mmx''.\nType ''help mmx'' for details.'));