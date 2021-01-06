%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

function [eta, B1,B3] = precond_laplace_overlapJacobi( L, xi, xL, xR, G, B1, B3 )
% L is a cell of operators

r = xi.rank;
n = xi.size;
d = xi.order;

% If B1 and B3 are not given as arguments, we need to precalculate them
if nargin < 7
%     % if applying L is expensive (not just tridiag), one can store all 
%     % applications with xL and compute the ones for xR with G.
%     % You need to first store LUl
%     LUl = cell(d,1);
%     for idx = 1:d
%         LUl{idx} = tensorprod( xL.U{idx}, L{idx}, 2 );
%     end
%     % and then change to LUr in the loop for B3 below
%     %         if idx+1==d
%     %              LUr = tensorprod( LUl{idx+1}, G{idx}, 1, true);
%     %         else
%     %             LUr = tensorprod( tensorprod( LUl{idx+1}, G{idx+1}', 3), G{idx}, 1, true);
%     %         end
    
    B1 = cell(d,1);
    B1{1} = 0;
    for idx = 2:d
        LUl = tensorprod( xL.U{idx-1}, L{idx-1}, 2 );
        if idx>2
            TT = tensorprod( xL.U{idx-1}, B1{idx-1}, 1 );
        else
            TT = 0;
        end
        B1{idx} = unfold(xL.U{idx-1},'left')'*unfold(TT + LUl,'left');
    end

    B3 = cell(d,1);
    for idx = d-1:-1:1
        LUr = tensorprod( xR.U{idx+1}, L{idx+1}, 2 );
        if idx<d-1
            TT = tensorprod( xR.U{idx+1}, B3{idx+1}, 3 );
        else
            TT = 0;
        end          
        B3{idx} = unfold(xR.U{idx+1},'right')*unfold(TT + LUr,'right')';
    end
    B3{d} = 0;
end

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


% 2. STEP: Solve ALS systems:

% B1 and B3 were precalculated before
for idx = 1:d
    rl = r(idx);
    rr = r(idx+1);
    
    B2 = L{idx};
  
    % Solve via the diagonalization trick
    [V1,E1] = eig(B1{idx}); [V3,E3] = eig(B3{idx});
    V = kron(V3,V1);
    EE = diag(E1)*ones(1,rr) + ones(rl,1)*diag(E3)'; E = EE(:);
    
    rhs = matricize( Y{idx}, 2 ) * V;
    Z = zeros(size(rhs));
    for i=1:length(E)
        Z(:,i) = (B2 + E(i)*speye(n(idx))) \ rhs(:,i);
    end
    eta.dU{idx} = tensorize( Z*V', 2, [rl, n(idx), rr] );
end

eta = TTeMPS_tangent_orth( xL, xR, eta );   % todo? Can we improve efficiency since eta is not a generic TTeMPS but shares the same x.U as xL and xR

end

