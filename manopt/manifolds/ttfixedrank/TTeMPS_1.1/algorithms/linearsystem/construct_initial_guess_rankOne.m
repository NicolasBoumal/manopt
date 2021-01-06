%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

function X = construct_initial_guess(L, F, r, n)
% Basicially only the first micro-step of ALS

X = TTeMPS_rand( r, n );
X = 1/norm(X) * X;


d = X.order;
n = X.size;


X = orthogonalize(X, 1);
Fi = contract( X, F, 1 );
sz = [X.rank(1), X.size(1), X.rank(2)];

[left, right] = Afun_prepare( L, X, 1 );
B1 =  prepare_precond( L.A{1}, X, 1 );

Ui = pcg( @(y) Afun( L, y, 1, sz, left, right), ...
         Fi(:), ...
         1e-10, 1000, ...
         @(y) apply_precond( L.A{1}, B1, y, sz ), [],...
         X.U{1}(:) ); 

X.U{1} = reshape( Ui, sz );

X = orth_at( X, 1, 'left', true );


end




function [left, right] = Afun_prepare( A, x, idx )
    y = A.apply(x); 
    if idx == 1
        right = innerprod( x, y, 'RL', idx+1 );
        left = [];
    elseif idx == x.order
        left = innerprod( x, y, 'LR', idx-1 );
        right = [];
    else
        left = innerprod( x, y, 'LR', idx-1 );
        right = innerprod( x, y, 'RL', idx+1 ); 
    end
end

function res = Afun( A, U, idx, sz, left, right )

    V = reshape( U, sz );
    V = A.apply( V, idx );
    
    if idx == 1
        tmp = tensorprod( V, right, 3 );
    elseif idx == A.order
        tmp = tensorprod( V, left, 1 );
    else
        tmp = tensorprod( V, right, 3);
        tmp = tensorprod( tmp, left, 1);
    end

    res = tmp(:);
end


function B1 = prepare_precond( L0, X, idx )

    if idx == 1
        B1 = [];
        return
    end

    n = size(L0, 1);
    r = X.rank;

    X1 = matricize( X.U{1}, 2);
    Y = X;
    Y.U{1} = tensorize( L0*X1, 2, [r(1), n(1), r(2)] );
    B1 = innerprod( X, Y, 'LR', idx-1);
end

function res = apply_precond( L0, B1, rhs, sz )
    
    n = size(L0, 1);
    rhs = reshape( rhs, sz );
    if isempty(B1) %idx == 1
        res = L0 \ unfold( rhs, 'left' );
        res = reshape( res, sz );
    else
        res = B1 \ unfold(rhs, 'right');
        res = reshape( res, sz );
    end
    res = res(:);
end






