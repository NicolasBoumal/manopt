function C = multiprod(A, B, unused1, unused2) %#ok<INUSD>
% Matrix multiply 2-D slices of N-D arrays
%
% function C = multiprod(A, B)
%
% If A, B are two 3-D arrays of compatible sizes, then C is a 3-D array
% such that
%
%     C(:, :, i) = A(:, :, i) * B(:, :, i)
%
% for each i. Arrays have compatible sizes if the above is well defined.
%
% If A, B are two N-D arrays of compatible sizes, then C is an N-D array
% such that
%
%     C(:, :, i, j, k, ...) = A(:, :, i, j, k, ...) * B(:, :, i, j, k, ...)
%
% for each i, j, k...
%
% This function is just a wrapper for pagemtimes, with a fallback call to
% multiprod_legacy in case pagemtimes is not available.
% If pagemtimes is available, it is better to call it directly. This is
% especially true if multiprod is called in conjunction with multitransp,
% as pagemtimes allows for both to be done in one function call.
%
% See also: multitransp multihconj multiscale multiskew multiskewh multitrace

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Aug. 12, 2021.
% Contributors: Xiaowen Jiang
% Change log: 
%
%   Aug. 12, 2021 (NB):
%       Matlab R2020b introduced a built-in function pagemtimes which does
%       essentially everything we ever needed to do with multiprod in
%       Manopt, and is much faster. It also has the advantage of being
%       compatible with dlarray, necessary for automatic differentiation.
%       Accordingly, multiprod became a wrapper for pagemtimes, and the old
%       code for multiprod remains available as multiprod_legacy.

    assert(nargin == 2, ...
           'The new multiprod only takes two inputs. Check multiprod_legacy.');

    if exist('pagemtimes', 'builtin') % Added to Matlab R2020b
        C = pagemtimes(A, B);
    else
    %   warning('manopt:multi', ...
    %          ['Matlab R2020b introduced pagemtimes, faster than multiprod.\n' ...
    %           'Calling the old code multiprod_legacy instead.\n' ...
    %           'To disable this warning: warning(''off'', ''manopt:multi'')']);
        C = multiprod_legacy(A, B);
    end

end
