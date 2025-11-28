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
%
%   Nov. 28, 2025 (NB):
%       Vanni Noferini pointed out that the call to exist(..., 'file') is
%       slow. The code was changed so that the check is executed only once.

    assert(nargin == 2, ...
           ['The new multiprod only takes two inputs. ' ...
            'Check multiprod_legacy.']);

    % Determine once whether pagemtimes is available.
    % It was added to Matlab R2020b.
    persistent haspagemtimes;
    if isempty(haspagemtimes)
        haspagemtimes = ~isempty(which('pagemtimes'));
    end

    if haspagemtimes
        C = pagemtimes(A, B);
    else
        C = multiprod_legacy(A, B);
    end

end
