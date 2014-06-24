%
% IN:
% 
% n m
% i1 j1 w_{i1,j1}
% i2 j2 w_{i2,j2}
%
% OUT:
%
% The sedumi formulation of 
% 
%  Min -C*X
%  st. X psd
%      X_{ii}      = 1 (i=1,..,n)
%
% where W = (w_{ij}), C = (1/4) * [ Diag(We) - W ]
% 
% The user must modify the file, according what the 
% I/O will be, at the positions marked with '!!!'
%
% After running 
% x = sedumi(A, b, c, K), you must take the 
% negative of the optimal solution to get the value listed in 
% the table 
%

vec = @(X) X(:);

% !!! 
infile   = fopen('torusg3-15.dat', 'r');

%
% Read data 
%

n = fscanf(infile, '%d', 1);
m = fscanf(infile, '%d', 1);
data = fscanf(infile, '%d %d %f', [3,m]);
data = data';

% !!!
% For the 'g' instances, (NOT for the 'pm' instances), 
% the edge-weights must be divided by 100,000 
%

data(:,3) = (1/100000)*data(:,3);

%
% Arrange data so that the first column is 
% always less than the second
% We check that edges are NOT given twice
% as both (i,j) and (j,i)
%

flipdata      = zeros( size(data,1), 2);
flipdata(:,1) = data(:,2);
flipdata(:,2) = data(:,1);

l = ( data(:,1)>data(:,2) );
data(l, 1:2) = flipdata(l, 1:2);


%
% Compute, and print some useful info
%

n_edges      = size(data,1) 
n_rows = n*(n+1)/2 
format long
sum_of_weights = sum(data( : ,3) )
posdata = data( data(:,3)>0, :);
sum_of_posweights = sum(posdata( : ,3) )

%
% Construct set ind_eq1
% 
% i*(n-1) + i is contained in the vector ind_eq1
% for i=1,..,n, since 
% X_{ii} = 1   is  a constraint 
%
 
ind_eq1 = ([ 0 : (n-1) ]*n + [ 1 : n ]);


%
% Construct C 
%

C = sparse( data(:,1), data(:,2), data(:,3), n, n );
C = C + C';
C = (1/2)*(diag(sum(C,1)) - C);

%
% Construct the constraints
%
% A*X = e
%
% where A  is n by n^2
%

A = sparse(  1:n, ind_eq1, ones(n,1) ); 
b = ones(n,1);
c = vec(-C)'; 

K.l = 0;
K.q = 0;
K.r = 0;
K.s = n;

%%% !!! 
save torusg3-15  A b c K 





  