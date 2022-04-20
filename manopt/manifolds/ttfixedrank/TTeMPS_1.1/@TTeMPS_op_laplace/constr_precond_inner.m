function expB = constr_precond_inner( A, X, mu )

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    n = size(A.L0, 1);
    sz = [X.rank(mu), X.size(mu), X.rank(mu+1)];

    B1 = zeros( X.rank(mu) );
    % calculate B1 part:
    for i = 1:mu-1
        % apply L to the i'th core
        tmp = X;
        tmp.U{i} = tensorprod_ttemps( tmp.U{i}, A.L0, 2 );
        B1 = B1 + innerprod( X, tmp, 'LR', mu-1);
    end

    % calculate B2 part:
    B2 = A.L0;

    B3 = zeros( X.rank(mu+1) );
    % calculate B3 part:
    for i = mu+1:A.order
        tmp = X;
        tmp.U{i} = tensorprod_ttemps( tmp.U{i}, A.L0, 2 );
        B3 = B3 + innerprod( X, tmp, 'RL', mu+1);
    end
    
    [V1,e1] = eig(B1);
    e1 = diag(e1);
    [V3,e3] = eig(B3);
    e3 = diag(e3);

    lmin = min(e1) + min(A.E_L) + min(e3);
    lmax = max(e1) + max(A.E_L) + max(e3);

    R = lmax/lmin;
    
    [omega, alpha] = load_coefficients( R );

    k = length(omega);
    omega = omega/lmin;
    alpha = alpha/lmin;

    expB = cell(3,k);
    
    for i = 1:k
        expB{1,i} = omega(i) * V1*diag( exp( -alpha(i)*e1 ))*V1';    % include omega in first part
        expB{2,i} = A.V_L*diag( exp( -alpha(i)*A.E_L ))*A.V_L';
        expB{3,i} = V3*diag( exp( -alpha(i)*e3 ))*V3';
    end
end

