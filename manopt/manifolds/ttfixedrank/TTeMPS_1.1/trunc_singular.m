function s = trunc_singular(s, tol, relative, maxrank)
% REL_TRUNC_SINGULAR Helper routine to truncate singular values

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

    if ~exist('relative','var'),    relative = true;     end
    if ~exist('maxrank','var'), maxrank = length(s); end

    summ = cumsum(s.^2,'reverse');

    if relative
        s = find(summ > tol^2, 1, 'last');
        if isempty(s), s = 1; end
    else
        s = find(summ > tol^2*summ(1), 1, 'last');
        if isempty(s), s = 1; end
    end

    s = min([s, maxrank, length(s)]);

end

