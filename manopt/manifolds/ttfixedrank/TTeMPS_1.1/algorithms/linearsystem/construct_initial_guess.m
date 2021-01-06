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
expB = constr_precond_inner( L, X, 1 );
Ui = pcg( @(y) Afun( L, y, 1, sz, left, right), ...
                 Fi(:), ...
                 1e-6, 1000, ...
                 @(y) apply_local_precond( L, y, sz, expB ), [],...
                 X.U{1}(:) ); 
                 %[], [], ...

X.U{1} = reshape( Ui, size(X.U{1}) );

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

function res = apply_local_precond( A, U, sz, expB)

    p = size(U, 2);

    x = reshape( U, [sz, p] );
    res = zeros( [sz, p] );

    for i = 1:size( expB, 1)
        tmp = reshape( x, [sz(1), sz(2)*sz(3)*p] );
        tmp = reshape( expB{1,i}*tmp, [sz(1), sz(2), sz(3), p] );

        tmp = reshape( permute( tmp, [2 1 3 4] ), [sz(2), sz(1)*sz(3)*p] );
        tmp = ipermute( reshape( expB{2,i}*tmp, [sz(2), sz(1), sz(3), p] ), [2 1 3 4] );

        tmp = reshape( permute( tmp, [3 1 2 4] ), [sz(3), sz(1)*sz(2)*p] );
        tmp = ipermute( reshape( expB{3,i}*tmp, [sz(3), sz(1), sz(2), p] ), [3 1 2 4] );

        res = res + tmp;
    end
    res = reshape( res, [prod(sz), p] );
    
end



