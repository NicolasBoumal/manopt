%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

function [eta,Yeta] = precond_laplace_overlapGS( L, xi, xL, xR, G, Lapl )
    
    r = xi.rank;
    n = xi.size;
    d = xi.order;

    eta = xi;
    xi = tangent_to_TTeMPS( xi );

    for idx=1:d
       eta.dU{idx} = eta.dU{idx}*0;
    end


    % 1. STEP: Project right hand side

     Y = cell(1,d);
    % contract to first core
    right = innerprod( xR, xi, 'RL', 2 );             
    Y{1} = tensorprod_ttemps( xi.U{1}, right, 3 );      

    for idx = 2:d-1
        left = innerprod( xL, xi, 'LR', idx-1 );
        right = innerprod( xR, xi, 'RL', idx+1 ); 
        res = tensorprod_ttemps( xi.U{idx}, left, 1 );
        Y{idx} = tensorprod_ttemps( res, right, 3 );
    end 
    
    % contract to last core
    left = innerprod( xL, xi, 'LR', d-1 );
    Y{d} = tensorprod_ttemps( xi.U{d}, left, 1 );

    % 2. STEP: Solve ALS systems:
    % Instead of doing  
    %    X_mid = orthogonalize(xR, idx); 
    % we recursively adjust the gauge based on xL and xR
    X_mid = xR;
    eta.dU{1} = solve_inner( L{1}, X_mid, Y{1}, 1 );
    for idx = 2:d-1
        X_mid.U{idx-1} = xL.U{idx-1};
        X_mid.U{idx} = tensorprod_ttemps(X_mid.U{idx},G{idx-1},1);
        
        Eta = tangent_to_TTeMPS( eta );
        
        PEta = xi - apply(Lapl, Eta);
        left = innerprod( xL, PEta, 'LR', idx-1 );
        right = innerprod( xR, PEta, 'RL', idx+1 ); 
        res = tensorprod_ttemps( PEta.U{idx}, left, 1 );
        Yeta = tensorprod_ttemps( res, right, 3 );
        
        
        eta.dU{idx} = solve_inner( L{idx}, X_mid, Yeta, idx );  
    end
    
    X_mid.U{d-1} = xL.U{d-1};
    X_mid.U{d} = tensorprod_ttemps(X_mid.U{d},G{d-1},1);

    Eta = tangent_to_TTeMPS( eta );

    PEta = xi - apply(Lapl, Eta);
    left = innerprod( xL, PEta, 'LR', d-1 ); 
    Yeta = tensorprod_ttemps( PEta.U{d}, left, 1 );

    eta.dU{d} = solve_inner( L{d}, X_mid, Yeta, d );  
    
    for idx = d-1:-1:2
        X_mid.U{idx+1} = xR.U{idx+1};
        X_mid.U{idx} = tensorprod_ttemps(X_mid.U{idx},(G{idx}'),3);
        
        Eta = tangent_to_TTeMPS( eta );
        
        PEta = xi - apply(Lapl, Eta);
        left = innerprod( xL, PEta, 'LR', idx-1 );
        right = innerprod( xR, PEta, 'RL', idx+1 ); 
        res = tensorprod_ttemps( PEta.U{idx}, left, 1 );
        Yeta = tensorprod_ttemps( res, right, 3 );

        eta.dU{idx} = eta.dU{idx} + solve_inner( L{idx}, X_mid, Yeta, idx );  
    end
    
    
    X_mid.U{1+1} = xR.U{1+1};
        X_mid.U{1} = tensorprod_ttemps(X_mid.U{1},(G{1}'),3);
        
        Eta = tangent_to_TTeMPS( eta );
        
        PEta = xi - apply(Lapl, Eta);
        right = innerprod( xR, PEta, 'RL', 2 ); 
        Yeta = tensorprod_ttemps( PEta.U{1}, right, 3 );  

        eta.dU{1} = eta.dU{1} + solve_inner( L{1}, X_mid, Yeta, 1 );  
        

        
    

    eta = TTeMPS_tangent_orth( xL, xR, eta );   
    % todo? we could improve efficiency since eta is not a generic TTeMPS but shares the same x.U as xL and xR

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

