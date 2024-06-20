function L = replacesparseentries(M, x, inplace)
% Creates a matrix with the same sparsity as M and values replaced by x.
%
% function L = replacesparseentries(M, x)
% function M = replacesparseentries(M, x, 'inplace')
%
% Inputs:
%   M: a sparse matrix with k nonzeros (nnz(M) = k)
%   x: a vector of k real doubles (numel(x) = k)
%   'inplace': if this string is passed as third input, then the entries of
%              M itself are replaced by those in x (regardless of nargout).
%
% Output:
%   L: a sparse matrix with the same sparsity pattern as M but with the
%      values replaced by the values in x, in the same order.
%
%   This is /almost/ the same as the following:
%
%       [m, n] = size(M);
%       k = nnz(M);
%       [i, j] = find(M);
%       L = sparse(i, j, x, m, n, k);
%
%   For efficiency, the operation is executed via a C-Mex function which
%   first copies M into L and then replaces the values in memory directly.
%   This avoids the need for Matlab to recompute the sparsity structure.
%   
%   Nuance: if x contains zeros, then this function keeps those zeros,
%   whereas sparse(...) would eliminate them from the sparsity structure.
%   Sometimes, the former is preferable to the latter.
%   
%   Pre-compiled files are included with Manopt.
%   To (re)compile for your own system, run
%       mex setsparseentries.c -largeArrayDims
%   in the folder that contains setsparseentries.c (that is, manopt/tools/)
%
% See also: sparseentrywisemult

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, June 7, 2024.
% Contributors:
% Change log:
    
    assert(issparse(M), 'M must be sparse.');

    assert(nnz(M) == numel(x), ...
           'x must contain as many numbers as the nonzeros in M.');
    
    assert(isa(x, 'double') && isreal(x) && ...
           isa(M, 'double') && isreal(M), ...
           'M and x must contain real, double floating points numbers.');
    
    % By default, we copy M to L and modify the entries of L.
    % Upon request, we can replace the entries of M directly.
    do_inplace = exist('inplace', 'var') && strcmp(inplace, 'inplace');
    if ~do_inplace
        % Copy the sparse matrix M into L.
        % Careful that Matlab uses copy-on-write:
        % https://stackoverflow.com/a/36062575/5989015
        L = M;          % This only makes L a reference to M.
        L(1) = L(1);    % This forces Matlab to actually copy M into L.
    
        % This C-Mex function /replaces/ the nonzero entries of L with x.
        % The code is in setsparseentries.c
        setsparseentries(L, x);
    else
        % Replace the entries of M directly.
        setsparseentries(M, x);
        L = M; % L points to M, but M itself was changed.
    end

end
