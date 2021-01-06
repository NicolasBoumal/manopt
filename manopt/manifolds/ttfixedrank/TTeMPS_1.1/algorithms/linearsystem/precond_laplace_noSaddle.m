%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

function [eta] = precond_laplace_noSaddle( L, xi, xL, xR )
    
    eta = xi;

    r = xi.rank;
    n = xi.size;
    d = xi.order;

    %L = repmat({L.L0}, 1, d); % for now, all L are the same

    % left side matrices. Solved later by Kronecker structure for core 2,... d
    A = cell(1, d);
    M = cell(1, d);
    A{1} = L{1};
    for i = 2:d
        M{i} = unfold( xi.U{i-1}, 'left')' * A{i-1} * unfold( xi.U{i-1}, 'left');
        A{i} = kron( speye(n(i)), M{i} ) + kron( L{i}, eye(r(i)) );
    end


    % right side matrices (to diagonalize), appearing for cores 1, ... d-1
    B = cell(1, d-1);
    B{d-1} = unfold( xi.V{d}, 'right' ) * kron( speye(r(d+1)), L{d} ) ...
                * unfold( xi.V{d}, 'right' )';

    for i = d-2:-1:1
        B{i} = unfold( xi.V{i+1}, 'right' ) * ...
                ( kron( speye(r(i+2)), L{i+1} ) + kron( B{i+1}, speye(n(i+1)) ) ) ...
                * unfold( xi.V{i+1}, 'right' )'; 
    end
    
    % calculate first core (special):
    [Q, lam] = eig(B{1}); lam = diag(lam);
    dU1l_eta = zeros( [n(1), r(2)] );
    U1lQ = unfold( xi.U{1}, 'left') * Q;
    dU1l_xi = unfold( xi.dU{1}, 'left' ) * Q; 
    for i = 1:r(2)
        dU1l_eta(:,i) = solve_saddle( A{1}, lam(i), U1lQ, dU1l_xi(:,i) );
    end
    eta.dU{1} = reshape( dU1l_eta*Q', size(xi.dU{1}) );        

    % calculate middle cores:
    for i = 2:d-1
        [Q, lam] = eig(B{i}); lam = diag(lam);
        dUl_eta = zeros( [r(i)*n(i), r(i+1)] );
        UQ = reshape( unfold( xi.U{i}, 'left') * Q, size(xi.U{i}) );
        dUQ_xi = reshape( unfold( xi.dU{i}, 'left') * Q, size(xi.dU{i}) );
        for j = 1:r(i+1)
            dUl_eta(:,j) = solve_saddle_fast( L{i}, M{i}, lam(j), ...
                                UQ, dUQ_xi(:,:,j));
        end
        eta.dU{i} = reshape( dUl_eta*Q', size(xi.dU{i}) );                        
    end

    % calculate last core (special):
    [Q, gam] = eig( M{d} );
    gam = diag(gam);
    eta.dU{d} = solve_kron( L{d}, 0, Q, gam, xi.dU{d} );

    
    eta = TTeMPS_tangent_orth( xL, xR, eta );
    
    
    
end


function res = solve_saddle( A, lam, Ul, rhs )

    As = (A + lam*speye(size(A)));
 
    res = As \ rhs;
    
end

function res = solve_saddle_fast( A, M, lam, U, rhs )

    [Q, gam] = eig(M);
    gam = diag(gam);
    
    
    % Step 4: res = A^{-1} * (rhs - Ul*y)
    d = unfold(rhs, 'left');
    d = reshape( d, size(rhs) );
    res = solve_kron( A, lam, Q, gam, d );
    res = unfold( res, 'left');
end


function sol = solve_kron( A, lam, Q, gam, rhs )
    
    if size(rhs, 3) == 1
        rhst_2 = rhs.' * Q; % matricize for vector == transpose
    else
        rhst_2 = matricize(rhs, 2) * kron( eye(size(rhs,3)), Q );
    end

    solt_2 = zeros(size(rhst_2));
    for i=1:length(gam)
        solt_2(:, i:length(gam):end) = ( A + (lam + gam(i))*speye(size(A)) ) ... 
                                                    \ rhst_2(:,i:length(gam):end);
    end

    if size(rhs, 3) == 1
        sol_2 = solt_2 * Q'; 
        sol = sol_2.'; % tensorize for vector == transpose
    else
        sol_2 = solt_2 * kron( eye(size(rhs,3)), Q' );
        sol = tensorize( sol_2, 2, size(rhs) );
    end

end

