function x = sparseentrywisemult(M, L, R)
% Computes the entrywise product of sparse M with L*R', without forming LR'
%
% function x = sparseentrywisemult(M, L, R)
%
% Inputs:
%   M: a sparse matrix of size mxn
%   L: a full matrix of size mxk
%   R: a full matrix of size nxk
%
% Output:
%   x = nonzeros(M.*(L*R.'))
%       This is a column vector matching the sparsity pattern of M.
%
%   Note that x is not computed as above.
%   Instead, the computation is done via a C-Mex function:
%   see sparseentries.
%
% See also: sparseentries

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, June 7, 2024.
% Contributors:
% Change log:

    x = nonzeros(M) .* sparseentries(M, L, R);

end
