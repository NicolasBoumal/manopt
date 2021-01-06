%   Rank-1 gradient approximation to increase the rank.
%   Unfortunately, not really worth the effort,
%   and a random rank-1 increase works equally well.
%
%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt
function X = increaseRank_mod( X, A_Omega, Omega, idx )

	r = X.rank;
	n = X.size;
	d = X.order;

	epsilon = 1;
	
	rankInc = 2;
	
	%idx = d-1;
	idxL = idx;
	idxR = idx + 1;

    X = orthogonalize(X, idxR);

	%Omega = sub2ind( n, deal(mat2cell(Omega, size(Omega,1), ones(1,d))) );
	Omega_ind = sub2ind( n, Omega(:,1), Omega(:,2), Omega(:,3), Omega(:,4), Omega(:,5) );
	Z = zeros( n );
	Z(Omega_ind) =  X(Omega) - A_Omega;
	
	% right side
	for i = d-1:-1:idxR
		zz = reshape( Z, [prod(n(1:i)), n(i+1)*r(i+2)] );
		xx = transpose( unfold( X.U{i+1}, 'right') );
		Z = zz*xx;
    end
    
	% left side
	for i = 1:idxL-1
		Z_i = reshape( Z, [r(i)*prod(n(i)), prod(n(i+1:idxR))*r(idxR+1)] );
		X_i = unfold( X.U{i}, 'left'); 
		Z = X_i' * Z_i;
    end
    size(Z)
	Z = reshape( Z, [r(idxL)*n(idxL), n(idxR)*r(idxR+1)] );

    % truncate to rank-1
	
	norm_Z = norm(Z(:))
	
	[U,S,V] = svd( Z, 'econ');
	Z1 = U(:,1:rankInc)*S(1:rankInc,1:rankInc);
	Z2 = V(:,1:rankInc)';
	
	Z1 = reshape( Z1, [r(idxL), n(idxL), rankInc] );
	Z2 = reshape( Z2, [rankInc, n(idxR), r(idxR+1)]);
	
	
	epsilon = fminbnd( @(t) linesearch(X, t, idxL, idxR, Z1, Z2, Omega, A_Omega), -100, 1)
	epsilon2 = linesearch2(X, idxL, idxR, Z1, Z2, Omega, A_Omega)
	
	X.U{idxL} = cat( 3, X.U{idxL}, epsilon2*Z1 );
	X.U{idxR} = cat( 1, X.U{idxR}, Z2 );
	
end


function res = linesearch( X, t, idxL, idxR, Z1, Z2, Omega, A_Omega )
    X.U{idxL} = cat( 3, X.U{idxL}, t*Z1 );
	X.U{idxR} = cat( 1, X.U{idxR}, Z2 );
	
	res = 0.5*norm( A_Omega - X(Omega) )^2;
end

function res = linesearch2( X, idxL, idxR, Z1, Z2, Omega, A_Omega )
    Y = X;
    Y.U{idxL} = Z1;
	Y.U{idxR} = Z2;
	
	res = Y(Omega)'*(A_Omega - X(Omega)) / norm(Y(Omega))^2;
end
