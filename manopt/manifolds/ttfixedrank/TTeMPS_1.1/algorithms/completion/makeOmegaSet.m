%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt
function subs = makeOmegaSet( n, sizeOmega )
    
    if sizeOmega > prod(n)
        error('makeOmegaSet:sizeOmegaTooHigh', 'Requested size of Omega is bigger than the tensor itself!')
    end

    idx = randi( prod(n), sizeOmega, 1 );
    Omega = unique(idx);

    while length(Omega) < sizeOmega
        idx = [ Omega; randi( prod(n) , sizeOmega-length(Omega), 1 )]; 
        Omega = unique(idx);
    end
    
    Omega = sort( Omega(1:sizeOmega) );

	% get number of dimensions
	d = length(n);
	% prepare dynamically sized varargout for ind2sub 
	% (careful! needs to be done because behaviour of ind2sub depends
	% on the number of output arguments)
    c = cell(1,d);
    [c{:}] = ind2sub( n, Omega );

    subs = [c{:}];
end
