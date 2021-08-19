function B = multitransp(A, unused) %#ok<INUSD>
% Transpose the matrix slices of an N-D array (no complex conjugate)
%
% function B = multitransp(A)
%
% If A is a 3-D array, then B is a 3-D array such that
%
%     B(:, :, i) = A(:, :, i).'
%
% for each i. If A is an N-D array, then B is an N-D array with the slices
% A(:, :, i, j, k, ...) transposed.
%
% This function is just a wrapper for pagetranspose, with a fallback call
% to multitransp_legacy in case pagetranspose is not available.
% If pagetranspose is available, it is better to call it directly.
% Note that pagemtimes also allows to compute products with transposes
% without explicitly transposing arrays.
%
% See also: multiprod multihconj multiscale multiskew multiskewh multitrace

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Aug. 12, 2021.
% Contributors: Xiaowen Jiang
% Change log: 
%
%   Aug. 12, 2021 (NB):
%       Matlab R2020b introduced a built-in function pagetranspose which
%       does essentially everything we ever needed to do with multitransp
%       in Manopt. Accordingly, multitransp became a wrapper for
%       pagetranspose, and the old code for multitransp remains available
%       as multitransp_legacy.

    assert(nargin == 1, ...
           'The new multitransp only takes one input. Check multitransp_legacy.');

    if exist('pagetranspose', 'file') % Added to Matlab R2020b
        B = pagetranspose(A);
    else
    %   warning('manopt:multi', ...
    %          ['Matlab R2020b introduced pagetranspose.\n' ...
    %           'Calling the old code multitransp_legacy instead.\n' ...
    %           'To disable this warning: warning(''off'', ''manopt:multi'')']);
        B = multitransp_legacy(A);
    end

end
