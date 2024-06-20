function x = sparseentries(M, L, R)
% Computes the entries of L*R.' matching the sparsity of M, w/o forming L*R.'
%
% function x = sparseentries(M, L, R)
%
% Inputs:
%   M: a sparse matrix of size mxn
%   L: a full matrix of size mxk
%   R: a full matrix of size nxk
%
% Output:
%   x = X(find(M)); with X = L*R.';
%       This is a column vector matching the sparsity pattern of M.
%
%   Note that x is not computed as above.
%   Instead, the computation is done via a C-Mex function with complexity
%   proportional to k x nnz(M). The product L*R.' is not computed.
%   Consequently, outputs may differ in proportion to machine precision.
%   
%   Pre-compiled files are included with Manopt.
%   To (re)compile for your own system, run
%       mex spmaskmult.c -largeArrayDims
%   in the folder that contains spmaskmult.c (that is, manopt/tools/)
%
% See also: sparseentrywisemult

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, June 7, 2024.
% Contributors:
% Change log:

    [I, J, ~] = find(M);
    
    assert(size(L, 1) == size(M, 1) && ...
           size(R, 1) == size(M, 2) && ...
           size(L, 2) == size(R, 2), ...
           'L*R.'' should have the same size as M.');
    
    assert(isa(L, 'double') && isa(R, 'double'), ...
           'L and R must contain double floating points.');
    
    assert(isreal(L) && isreal(R), ...
           'L and R must contain real numbers.');
    
    % This is a C-Mex function, the code is in spmaskmult.c
    x = spmaskmult(L, R.', uint32(I), uint32(J));

end
