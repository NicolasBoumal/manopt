function res = mergecore( x, idxL, idxR )
    %MERGECORE Merging of two cores of a TT/MPS tensor.
    %
    %   RES = MERGECORE(X, IDXL, IDXR) merges the two cores IDXL and IDXR of
    %   the TT/MPS tensor X. IDXL and IDXR must be two consecutive integers in 
    %   ascending order. RES is a TT/MPS tensor of dimension X.order-1.
    %
    %   See also SPLITCORE

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt
    
    if ~isscalar(idxL)
         error('Left index IDXL must be a scalar.')
    end 
    if ~isscalar(idxR)
         error('Right index IDXR must be a scalar.')
    end 
    if diff([idxL,idxR]) ~= 1
        error('Choose two neighboring nodes in ascending order.')
    end
        
    n = x.size;
    r = x.rank;
    
    mergedcore = unfold(x.U{idxL}, 'left') * unfold(x.U{idxR}, 'right');
    mergedcore = reshape( mergedcore, [r(idxL), n(idxL)*n(idxR), r(idxR+1)] ); 
    
    U = x.U;
    res = TTeMPS( {U{1:idxL-1}, mergedcore, U{idxR+1:end} } );
end
    
