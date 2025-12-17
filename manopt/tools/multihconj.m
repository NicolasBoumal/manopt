function B = multihconj(A, unused) %#ok<INUSD>
% Hermitian-conjugate transpose the matrix slices of an N-D array
%
% function B = multihconj(A)
%
% If A is a 3-D array, then B is a 3-D array such that
%
%     B(:, :, i) = A(:, :, i)'
%
% for each i. If A is an N-D array, then B is an N-D array with the slices
% A(:, :, i, j, k, ...) Hermitian-conjugate transposed.
%
% This function is just a wrapper for pagectranspose, with a fallback call
% to multihconj_legacy in case pagectranspose is not available.
% If pagectranspose is available, it is better to call it directly.
% Note that pagemtimes also allows to compute products with (c)transposes
% without explicitly (c)transposing arrays.
%
% See also: multiprod multitransp multiscale multiskew multiskewh multitrace

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Aug. 12, 2021.
% Contributors: Xiaowen Jiang
% Change log: 
%
%   Aug. 12, 2021 (NB):
%       Matlab R2020b introduced a built-in function pagectranspose which
%       does essentially everything we ever needed to do with multihconj
%       in Manopt. Accordingly, multihconj became a wrapper for
%       pagectranspose, and the old code for multihconj remains available
%       as multihconj_legacy.
%
%   Nov. 28, 2025 (NB):
%       Vanni Noferini pointed out that the call to exist(..., 'file') is
%       slow. The code was changed so that the check is executed only once.

    assert(nargin == 1, ...
           ['The new multihconj only takes one input. ' ...
            'Check multihconj_legacy.']);

    % Determine once whether pagectranspose is available.
    % It was added to Matlab R2020b.
    persistent haspagectranspose;
    if isempty(haspagectranspose)
        haspagectranspose = ~isempty(which('pagectranspose'));
    end

    if haspagectranspose
        B = pagectranspose(A);
    else
        B = multihconj_legacy(A);
    end

end
