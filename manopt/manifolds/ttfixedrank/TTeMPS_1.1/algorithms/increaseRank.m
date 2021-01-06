% Helper routine to artificially increase the rank of the current iterate.

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

function X = increaseRank( X, inc, idx, epsilon )

	r = X.rank;
	n = X.size;
	d = X.order;
	
	if ~exist('epsilon','var')
        epsilon = 1e-8;
    end

	X.U{idx-1} = cat( 3, X.U{idx-1}, epsilon*randn(r(idx-1), n(idx-1), inc) );
	X.U{idx} = cat( 1, X.U{idx}, epsilon*randn(inc, n(idx), r(idx+1)) );
end
