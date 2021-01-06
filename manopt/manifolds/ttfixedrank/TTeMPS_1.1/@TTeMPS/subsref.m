function out = subsref(x, s)
	%SUBSREF Subscripted reference for TTeMPS.
	%
	%   Examples:
	%   x([2,3,1]) computes and returns the element (2,3,1) of x.
	%   x( ind ) computes and returns all elements x(ind(i,:)) for i = 1:size(ind,1).
	%   x(:) returns the vectorization of full(x) (careful!)
    %   x{i} returns the i-th core of x, shorthand for x.U{i}
	%
	%   See also TTEMPS.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt
    

	switch s(1).type

	case '.'

		prop = properties('TTeMPS');
		if( any(strcmp(s(1).subs, prop) ) ),
		    out = builtin('subsref', x, s);
		else
			ll = length(prop);
			proplist = repmat({', '}, 2*ll-1, 1);
			proplist(1:2:end) = prop;
			proplist = cat(2,proplist{:});
			error(['Object TTeMPS does not have field ' s(1).subs ...
			'. The following fields are available: ' proplist '.']);
		end

	case '()'
		% x(:)
		ind = s(1).subs{1};
		if(length(ind) == 1) && (ind == ':')
			out = full(x);
			out = out(:);
			% e.g. x( [1,2,3; 4 5 6; 7 8 9; 3 5 3]) for d = 3 tensor
		elseif(size(ind,2) == x.order)
			
			r = x.rank;
			%out = zeros(size(ind,1), 1);
			
			%C = cell(1,x.order);
			%for i=1:x.order
			%	C{i} = permute( x.U{i}, [1 3 2]);
			%end
			%for i = 1:size(ind,1)
			%	p = C{1}(:,:,ind(i,1)); 			
			%	for j = 2:size(ind,2)
			%		p = p * C{j}(:,:,ind(i,j));
			%	end
			%	out(i) = p;
			%end
			n = x.size;
			
			C = cell(1,x.order);
			for i=1:x.order
				C{i} = permute( x.U{i}, [1 3 2]);
				C{i} = unfold( C{i}, 'right');
			end
			
            out = TTeMPS.subsref_mex( n, r, transpose(ind), C);

			%for i = 1:size(ind,1)
			%	p = C{1}(:, (ind(i,1)-1)*r(2)+1:ind(i,1)*r(2)); 			
			%	for j = 2:size(ind,2)
			%		p = p * C{j}(: , (ind(i,j)-1)*r(j+1)+1:ind(i,j)*r(j+1));
			%	end
			%	out(i) = p;
			%end
					
	
		else
			error('Number of indices does not match order of TTeMPS tensor.');
		end


	case '{}'

		if(length(s(1).subs) ~= 1 || ~isnumeric(s(1).subs{1}) || ...
			s(1).subs{1} <= 0)
			error('{} only takes one positive integer.');
		end

		ii = s(1).subs{1};
		if(ii > x.order)
			error('Index exceeds number of dimensions');
		end
		out = builtin('subsref', x.U, s);

	end
