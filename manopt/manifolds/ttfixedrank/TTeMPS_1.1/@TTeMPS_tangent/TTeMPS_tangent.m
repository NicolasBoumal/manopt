classdef TTeMPS_tangent
% TTeMPS_tangent
%
%   A MATLAB class for representing and tangent tensors
%   to the TT/MPS format. 
% 

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt


properties( SetAccess = public, GetAccess = public )

    dU
	
end

methods( Access = public )

    function Y = TTeMPS_tangent(X, Z, ind, vec)
    %TTEMPS_TANGENT projects the d-dimensional array Z into the tangent space at a TT/MPS tensor X.
    %
    %   P = TTEMPS_TANGENT(X) projects the d-dimensional array Z into the tangent space of the 
	%	TT-rank-r manifold at a TT/MPS tensor X.
	%
	%   Important: X has to be left-orthogonalized! (use X = orthogonalize(X, X.order) beforehand)
    %
        if nargin == 1
            % Create a zero tangent tensor
            Y.dU = cellfun( @(x) zeros(size(x)), X.U, 'UniformOutput', false );
            return
        end


		if ~exist('ind','var')
			sampled = false;
		else
			sampled = true;
            if ~exist('vec','var')
                vec = false;
            end
		end
		
		d = X.order;
		r = X.rank;
		n = X.size;
	
		invXtX = cell(1,d);
		tmp = conj(unfold( X.U{d}, 'right')) * unfold( X.U{d}, 'right').';
		invXtX{d} = pinv(tmp,1e-8);
		for i = d-1:-1:2     
		    tmp = tensorprod_ttemps( X.U{i}, tmp', 3);
		    tmp = conj(unfold( tmp, 'right')) * unfold( X.U{i}, 'right').';
			invXtX{i} = pinv(tmp,1e-8);
		end

		if isa(Z, 'TTeMPS')
			
			Y.dU = cell(1, d);
			
			for i = 1:d
				Y.dU{i} = contract( X, Z, i);
			end
			
			for i = 1:d-1
				Y.dU{i} = unfold( Y.dU{i}, 'left' );
				Y.dU{i} = Y.dU{i} * invXtX{i+1};
				
				%U = orth(unfold(X.U{i},'left')); % orth unnecessary
				U = unfold(X.U{i},'left');
				Y.dU{i} = Y.dU{i} - U * ( U' * Y.dU{i});
				Y.dU{i} = reshape( Y.dU{i}, [r(i), n(i), r(i+1)] );
			end
			
		elseif isa(Z, 'TTeMPS_tangent')
			
			Znew = tangent_to_TTeMPS( Z, X );
			Y = TTeMPS_tangent( X, Znew );
		
		elseif ~sampled % Z is a full array
		    
			ZZ = cell(1,d);

			% right side
			ZZ{d} = Z(:);
			for i = d-1:-1:1
				zz = reshape( Z, [prod(n(1:i)), n(i+1)*r(i+2)] );
				xx = transpose( unfold( X.U{i+1}, 'right') );
				Z = zz*xx;
				ZZ{i} = Z * invXtX{i+1};
			end

			% left side
			for k = 2:d
				for i = 1:k-1
					Z_i = reshape( ZZ{k}, [r(i)*prod(n(i)), prod(n(i+1:k))*r(k+1)] );
					X_i = unfold( X.U{i}, 'left');
					Z = X_i' * Z_i;
					ZZ{k} = Z;
				end
				ZZ{k} = reshape( ZZ{k}, [r(k)*n(k), r(k+1)] );
			end

			Y.dU = cell(1,d);
			% orth. projection (w/o last core)
			for i = 1:d-1
				U = orth(unfold(X.U{i},'left'));
				ZZ{i} = ZZ{i} - U * ( U' * ZZ{i});
				Y.dU{i} = reshape( ZZ{i}, [r(i), n(i), r(i+1)] );
			end
			Y.dU{d} = reshape( ZZ{d}, [r(d), n(d), r(d+1)] );
		
		else % Z is a sparse array
			
			vals = Z;
			C = cell(1,d);
			Y.dU = cell(1,d);
			for i=1:d
				C{i} = permute( X.U{i}, [1 3 2]);
				%Y.dU{i} = zeros(size(C{i}));
			end
			res = TTeMPS_tangent_omega( n, r, C, ind.', vals);
			
			for i = 1:d
				res{i} = reshape( res{i}, [r(i), r(i+1), n(i)] );
				Y.dU{i} = unfold( permute( res{i}, [1 3 2]), 'left');
			end
			
			for i=1:d-1
				Y.dU{i} = Y.dU{i} * invXtX{i+1};	
			end
			
			for i=1:d-1
				%U = orth(unfold(X.U{i},'left'));
				U = unfold(X.U{i},'left');
				Y.dU{i} = Y.dU{i} - U * ( U' * Y.dU{i});	
				Y.dU{i} = reshape( Y.dU{i}, [r(i), n(i), r(i+1)] );		
			end
			Y.dU{d} = reshape( Y.dU{d}, [r(d), n(d), r(d+1)] );
			
		end
	
		
	end
	
	function Zfull = full( Z, X )
	%FULL converts tangent tensor to d-dimensional array
    %
    %   ZFULL = full(Z, X) converts the tangent tensor Z given in factorized form
	%   (class TTeMPS_tangent) to a d-dimensional array ZFULL. X is the TTeMPS tensor at 
	%	which point the tangent space is taken.
    %
		d = X.order;
		
		C = cell(1, d);

		C{1} = cat( 3, Z.dU{1}, X.U{1} );
		for i = 2:d-1
			zeroblock = zeros( size(X.U{i}) );
			tmp1 = cat( 3, X.U{i}, zeroblock );
			tmp2 = cat( 3, Z.dU{i}, X.U{i} );
			C{i} = cat( 1, tmp1, tmp2 );
		end
		C{d} = cat( 1, X.U{d}, Z.dU{d} );
		
		Xnew = TTeMPS( C );
		Zfull = full( Xnew );
	end
	
	function res = plus( X, Y)
		%PLUS adds two tangent tensors
		%	    
	    %   RES = plus(X, Y) adds two tangent tensors in factorized form. Both tangent tensors
		%   have be elements of the SAME tangent space.
	    %
	
		res = X;
		res.dU = cellfun(@plus, X.dU, Y.dU, 'UniformOutput', false);
	end
	
	function X = minus( X, Y )
		%MINUS substracts two tangent tensors
		%	    
	    %   RES = minus(X, Y) substracts two tangent tensors in factorized form. Both tangent tensors
		%   have be elements of the SAME tangent space.
	    %
	
		X.dU = cellfun(@plus, X.dU, Y.dU, 'UniformOutput', false);
	end
	
	function X = mtimes( a, X )
		%MTIMES Multiplication of TTeMPS tangent tensor by scalar
		%	    
	    %   RES = mtimes(a, X) multiplies the TTeMPS tangent tensor X
	    %	by the scalar a.
		%
		
		X.dU = cellfun(@(x) a*x, X.dU, 'UniformOutput', false);	
	end
	function X = uminus( X )
		%UMINUS Unary minus of TTeMPS tangent tensor.
		%	    
	    %   RES = uminus(X) negates the TTeMPS tangent tensor X.
		%
		
		X = mtimes( -1, X );
	end
	
	function Xnew = tangentAdd( Z, t, X, retract )
	%TANGENTADD adds a tangent vector to a point on the manifold
    %	    
    %   RES = tangentAdd(Z, t, X) adds a tangent vector Z to a point X on the rank-r-manifold, scaled by t:
	% 			res = X + t*Z
	% 	where the result is stored as a TTeMPS tensor of rank 2*r.
	%	
	%   RES = tangentAdd(Z, t, X, true) adds a tangent vector Z to a point X on the rank-r-manifold, scaled by t:
	% 			res = X + t*Z
	%	and retracts the result back to the manifold:
	%			res = R_X( X + t*Z )
	%	where the result is stored as a TTeMPS tensor of rank r.
	%
	
	    if ~exist( 'retract', 'var' )
	        retract = false;
	    end

		d = X.order;
		C = cell(1, d);

		C{1} = cat( 3, t*Z.dU{1}, X.U{1} );
		for i = 2:d-1
			zeroblock = zeros( size(X.U{i}) );
			tmp1 = cat( 3, X.U{i}, zeroblock );
			tmp2 = cat( 3, t*Z.dU{i}, X.U{i} );
			C{i} = cat( 1, tmp1, tmp2 );
		end
		C{d} = cat( 1, X.U{d}, t*Z.dU{d} + X.U{d} );
		
		Xnew = TTeMPS( C );
	
		if retract
			Xnew = truncate( Xnew, X.rank );
		end
		
	end
    
	function res = innerprod( Z1, Z2, X )
		
		X1 = tangent_to_TTeMPS( Z1, X );
		X2 = tangent_to_TTeMPS( Z2, X );
		
		res = innerprod( X1, X2 );
	end
	
	function res = at_Omega( Z, ind, X)
		
		Xnew = tangent_to_TTeMPS( Z, X );
		res = Xnew(ind);
	end
	
	function res = tangent_to_TTeMPS( Z, X)
		d = X.order;
		C = cell(1, d);

		C{1} = cat( 3, Z.dU{1}, X.U{1} );
		for i = 2:d-1
			zeroblock = zeros( size(X.U{i}) );
			tmp1 = cat( 3, X.U{i}, zeroblock );
			tmp2 = cat( 3, Z.dU{i}, X.U{i} );
			C{i} = cat( 1, tmp1, tmp2 );
		end
		C{d} = cat( 1, X.U{d}, Z.dU{d});
		
		res = TTeMPS( C );
	end
end
	
	
	
    
end
