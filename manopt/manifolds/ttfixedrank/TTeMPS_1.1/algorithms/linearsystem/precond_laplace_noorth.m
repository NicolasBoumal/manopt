%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt
function [eta] = precond_laplace_noorth( L, xi, xL, xR, G )
    
    r = xi.rank;
    n = xi.size;
    d = xi.order;

    eta = xi;
    xi = tangent_to_TTeMPS( xi );

    % 1. STEP: Project right hand side

    Y = cell(1,d);
    % precompute inner products
    left = innerprod( xL, xi, 'LR', d-1, true );             
    right = innerprod( xR, xi, 'RL', 2, true );             

    % contract to first core
    Y{1} = tensorprod_ttemps( xi.U{1}, right{2}, 3 );      
    % contract to first core
    for idx = 2:d-1
        res = tensorprod_ttemps( xi.U{idx}, left{idx-1}, 1 );
        Y{idx} = tensorprod_ttemps( res, right{idx+1}, 3 );
    end 
    % contract to last core
    Y{d} = tensorprod_ttemps( xi.U{d}, left{d-1}, 1 );

    % 2. STEP: Solve ALS systems:
    % Instead of doing  
    %    X_mid = orthogonalize(xR, idx); 
    % we recursively adjust the gauge based on xL and xR
    X_mid = xR;
    eta.dU{1} = solve_inner( L{1}, X_mid, Y{1}, 1 );
    for idx = 2:d
        X_mid.U{idx-1} = xL.U{idx-1};
        X_mid.U{idx} = tensorprod_ttemps(X_mid.U{idx},G{idx-1},1);
        eta.dU{idx} = solve_inner( L{idx}, X_mid, Y{idx}, idx );  
    end

    eta = TTeMPS_tangent_orth( xL, xR, eta );   % todo? Can we improve efficiency since eta is not a generic TTeMPS but shares the same x.U as xL and xR

end

function X = get_mid(xL, xR, G, idx)
X = xR;
X.U{1:idx-1} = xL.U{1:idx-1};
if idx>1
    X.U{idx} = tensorprod_ttemps(X.U{idx},G{idx-1},1);
end
end

function [res,BB1,BB3] = solve_inner( L0, X, Fi, idx )
    n = size(L0, 1);
    rl = X.rank(idx);
    rr = X.rank(idx+1);

    B1 = zeros( rl );
    %BB1 = {};
    % calculate B1 part:
    for i = 1:idx-1
        % apply L to the i'th core
        tmp = X;
        tmp.U{i} = tensorprod_ttemps( tmp.U{i}, L0, 2 );
        %BB1{i} = tmp.U{i};
        B1 = B1 + innerprod( X, tmp, 'LR', idx-1);
    end

    % calculate B2 part:
    B2 = L0;

    B3 = zeros( rr );
    %BB3 = {};
    % calculate B3 part:
    for i = idx+1:X.order
        tmp = X;
        tmp.U{i} = tensorprod_ttemps( tmp.U{i}, L0, 2 );
        %BB3{i} = innerprod( X, tmp, 'RL', idx+1);
        B3 = B3 + innerprod( X, tmp, 'RL', idx+1);
    end

    % Faster below
    %[V,E] = eig( kron( eye(rr), B1 ) + kron( B3, eye(rl) ) );
    %E = diag(E);
            
    [V1,E1] = eig(B1); [V3,E3] = eig(B3);
    V = kron(V3,V1);    
    EE = diag(E1)*ones(1,rr) + ones(rl,1)*diag(E3)'; E = EE(:);
    
    rhs = matricize( Fi, 2 ) * V;
    Y = zeros(size(rhs));
    for i=1:length(E)        
        Y(:,i) = (B2 + E(i)*speye(n)) \ rhs(:,i);
    end
    res = tensorize( Y*V', 2, [rl, n, rr] );
end

