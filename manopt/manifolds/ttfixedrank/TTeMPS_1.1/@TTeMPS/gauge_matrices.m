function [xL, xR, G] = gauge_matrices( x )
% GAUGE_MATRICES Right and left orthogonalization with storage of gauge matrices
%
%   [xL,xR,G] = GAUGE_MATRICES(x) Compute a left and right orthogonalization and 
%   keep the gauge matrices that relates them.
%
%   The i-th core of xR
%        unfold(xR.U{i},'left')
%   is equal to the transformed i-th core of xL
%        kron(eye(n(i)),inv(G{i-1}))*unfold(xL.U{i},'left')*G{i}
%   (where fore i=1 and i=d, G{i} = 1).
%
%   Or, equivalently
%        tensorprod( tensorprod( xL.U{i}, G{i}', 3), inv(G{i-1}), 1)
%   equals
%        xR.U{i}.
%
%   See also LEFT_ORTH_WITH_GAUGE

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

    xR = orthogonalize( x, 1 );
    
    [xL, G] = left_orth_with_gauge( xR ); 
    
end
