%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

function [eta] = precond_rankOne( A, xi, xL, xR  )
% A is a parameterdependent operator

r = xL.rank;
n = xL.size;
d = xL.order;


eta = xi;
xi = tangent_to_TTeMPS( xi );
% % 1. STEP: Project right hand side
% below is hard-coded version of
% for ii=1:d
%     eta_partial_ii = TTeMPS_partial_project_overlap( xL, xR, xi, ii);
%     Y{ii} = eta_partial_ii.dU{ii};
% end

% TODO, it seems that the left and right cell arrays consist of a lot of
% identities and zeros.
Y = cell(1,d);
% precompute inner products
left = innerprod( xL, xi, 'LR', d-1, true );
right = innerprod( xR, xi, 'RL', 2, true );

% contract to first core
Y{1} = tensorprod( xi.U{1}, right{2}, 3 );
% contract to first core
for idx = 2:d-1
    res = tensorprod( xi.U{idx}, left{idx-1}, 1 );
    Y{idx} = tensorprod( res, right{idx+1}, 3 );
end
% contract to last core
Y{d} = tensorprod( xi.U{d}, left{d-1}, 1 );


tmp = (19.4*A.A{1}) \ unfold( Y{1}, 'left' );
eta.dU{1} = reshape( tmp, [r(1), n(1), r(2)]);
 
x1 = matricize( xL.U{1}, 2);
y = xL;
y.U{1} = tensorize( (19.4*A.A{1})*x1, 2, [r(1), n(1), r(2)] );
B1 = innerprod( xL, y, 'LR', d-1, true);
for i=2:d
    tmp = B1{i-1} \ unfold(Y{i}, 'right');
    eta.dU{i} = reshape( tmp, [r(i), n(i), r(i+1)] );
end

eta = TTeMPS_tangent_orth( xL, xR, eta ); 

end

