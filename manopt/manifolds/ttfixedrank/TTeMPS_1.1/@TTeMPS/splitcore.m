function res = splitcore( x, idx, nL, nR, tol )
    %SPLITCORE Merging of two cores of a TT/MPS tensor.
    %
    %   RES = SPLITCORE(X,IDX,NL,NR) splits the core IDX of
    %   the TT/MPS tensor X into two cores with outer dimensions NL and NR.
    %   The new outer dimensions NL and NR have to fulfill NL*NR = X.size(idx)
    %
    %   See also MERGECORE

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt
    
    if ~exist('tol', 'var')
        tol = 1e-8;
    end

    if ~isscalar(idx)
         error('Index IDX must be a scalar.')
    end 
    
    n = x.size;
    r = x.rank;
    
    if nL*nR ~= n(idx)
        error('New sizes must be compatible with old tensor: NL*NR = X.size(idx)')
    end
    
    %x = orthogonalize(x, idx);
    [U,S,V] = svd( reshape(x.U{idx}, [r(idx)*nL, nR*r(idx+1)]), 'econ');
    s = trunc_singular( diag(S), tol, true );
    U = U(:,1:s);
    V = V(:,1:s);
    S = S(1:s,1:s);
    newcoreR = reshape( S*V', [s, nR, r(idx+1)] );
    newcoreL = reshape( U, [r(idx), nL, s] );
    
    C = x.U;
    res = TTeMPS( {C{1:idx-1}, newcoreL, newcoreR, C{idx+1:end} } );
end
    
