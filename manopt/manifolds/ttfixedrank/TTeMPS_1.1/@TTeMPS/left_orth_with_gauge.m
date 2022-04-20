function [xL, G] = left_orth_with_gauge( xR )
% LEFT_ORTH_WITH_GAUGE Left orthogonalization with storage of gauge matrices
%
% Given a right orthogonal X, compute a left orthogonalization and keep 
% the gauge matrices that relates them.
%
%  The i-th core of xR
%       unfold(xR.U{i},'left')
%  is equal to the transformed i-th core of xL
%       kron(eye(n(i)),inv(G{i-1}))*unfold(xL.U{i},'left')*G{i}
%  (where fore i=1 and i=d, G{i} = 1).
%
%  Or, equivalently
%       tensorprod_ttemps( tensorprod_ttemps( xL.U{i}, G{i}', 3), inv(G{i-1}), 1)
%  equals
%       xR.U{i}.
%
%   See also GAUGE_MATRICES

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt
    
    xL = xR;
    G = cell(xR.order-1, 1);
    % left orthogonalization till pos (from left)
    for i = 1:xR.order-1
        [xL, G{i}] = orth_at( xL, i, 'left' );
    end
    
end
