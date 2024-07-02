%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

rng(11);

disp('Number of dimensions:')
d = 5

disp('Define tensor size:')
n = [7 8 9 10 11]

disp('Define rank vector (note that in the TT/MPS format, both the first and last rank are 1):')
r = [1 4 5 6 7 1]

disp('Create two random TT/MPS tensors:')
X = TTeMPS_rand( r, n )
Y = TTeMPS_rand( r, n )

disp('Calculate inner product between them:')
ip = innerprod( X, Y )

disp('Left-orthogonalize X:')
X = orthogonalize( X, 1 )

disp('Right-orthogonalize Y:')
Y = orthogonalize( Y, Y.order )

disp('Add X and Y')
Z = X + Y

disp('Truncate Z back to rank r:')
Z_trunc = truncate(Z, r)

disp('Note that we also have the round() operations,')
disp('where you specify a desired accuracy instead of prescribed rank')
Z_round = round(Z, 1e-2)
