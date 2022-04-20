%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt
function [X, residuum, cost, times] = alsLinsolve_fast( L, F, X, opts )

t_start = tic();
% set default opts
if ~exist( 'opts', 'var');       opts = struct();       end
if ~isfield( opts, 'nSweeps');   opts.nSweeps = 4;      end
if ~isfield( opts, 'solver');    opts.solver = 'pcg';   end

d = X.order;
n = X.size;


normF = norm(F);
g = apply(L, X) - F;
cost = cost_function_res( X, g );
residuum = norm( g ) / normF;
times = toc(t_start);

X = orthogonalize(X, 1);
for sweep = 1:opts.nSweeps
    % ====================================================================
    % LEFT-TO-RIGHT SWEEP
    % ====================================================================
    disp( ['STARTING SWEEP ', num2str(sweep), ' from left to right'] )
    disp( '===========================================================')
    for idx = 1:d-1
        disp( ['Current core: ', num2str(idx)] )

        Fi = contract( X, F, idx );
        sz = [X.rank(idx), X.size(idx), X.rank(idx+1)];
        
        if strcmpi( opts.solver, 'direct' )
            % if system very small
            Li = contract( L, X, idx );
            Ui = Li \ Fi(:);
            X.U{idx} = reshape( Ui, sz );

        elseif strcmpi( opts.solver, 'pcg' )

            [left, right] = Afun_prepare( L, X, idx );
            [B2, V, E] =  prepare_precond( L.L0, X, idx );

            Ui = pcg( @(y) Afun( L, y, idx, sz, left, right), ...
                     Fi(:), ...
                     1e-10, 1000, ...
                     @(y) apply_precond( B2, V, E, y, sz ), [],...
                     X.U{idx}(:) ); 

            X.U{idx} = reshape( Ui, sz );

        elseif strcmpi( opts.solver, 'diag' )
            X.U{idx} = solve_inner( L.L0, X, Fi, idx );

        else
            error( 'Unknown opts.solver type. Use either ''direct'', ''pcg'' (default) or ''diag''.' )
        end

        X = orth_at( X, idx, 'left', true );
        
        g = apply(L, X) - F;
        residuum = [residuum; norm( g ) / normF];
        cost = [cost; cost_function_res( X, g )];
        times = [times; toc(t_start)];
    end

    % ====================================================================
    % RIGHT-TO-LEFT
    % ====================================================================
    disp( 'Starting right-to-left half-sweep:')
    for idx = d:-1:2
        disp( ['Current core: ', num2str(idx)] )

        Fi = contract( X, F, idx );
        sz = [X.rank(idx), X.size(idx), X.rank(idx+1)];
        
        if strcmpi( opts.solver, 'direct' )
            % if system very small
            Li = contract( L, X, idx );
            Ui = Li \ Fi(:);
            X.U{idx} = reshape( Ui, sz );

        elseif strcmpi( opts.solver, 'pcg' )

            [left, right] = Afun_prepare( L, X, idx );
            [B2, V, E] =  prepare_precond( L.L0, X, idx );

            Ui = pcg( @(y) Afun( L, y, idx, sz, left, right), ...
                     Fi(:), ...
                     1e-10, 1000, ...
                     @(y) apply_precond( B2, V, E, y, sz ), [],...
                     X.U{idx}(:) ); 



            X.U{idx} = reshape( Ui, sz );

        elseif strcmpi( opts.solver, 'diag' )
            X.U{idx} = solve_inner( L.L0, X, Fi, idx );

        else
            error( 'Unknown opts.solver type. Use either ''direct'', ''pcg'' (default) or ''diag''.' )
        end


        X = orth_at( X, idx, 'right', true );
        
        g = apply(L, X) - F;
        residuum = [residuum; norm( g ) / normF];
        cost = [cost; cost_function_res( X, g )];
        times = [times; toc(t_start)];
    end
    
end


end

function res = cost_function( L, X, F )
res = 0.5*innerprod( X, apply(L, X) ) - innerprod( X, F );
end

function res = cost_function_res( X, res )
res = 0.5*innerprod( X, res );
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
        tmp = tensorprod_ttemps( V, right, 3 );
    elseif idx == A.order
        tmp = tensorprod_ttemps( V, left, 1 );
    else
        tmp = tensorprod_ttemps( V, right, 3);
        tmp = tensorprod_ttemps( tmp, left, 1);
    end

    res = tmp(:);
end

%function res = apply_local_precond( A, U, sz, expB)
%
%    p = size(U, 2);
%
%    x = reshape( U, [sz, p] );
%    res = zeros( [sz, p] );
%
%    for i = 1:size( expB, 1)
%        tmp = reshape( x, [sz(1), sz(2)*sz(3)*p] );
%        tmp = reshape( expB{1,i}*tmp, [sz(1), sz(2), sz(3), p] );
%
%        tmp = reshape( permute( tmp, [2 1 3 4] ), [sz(2), sz(1)*sz(3)*p] );
%        tmp = ipermute( reshape( expB{2,i}*tmp, [sz(2), sz(1), sz(3), p] ), [2 1 3 4] );
%
%        tmp = reshape( permute( tmp, [3 1 2 4] ), [sz(3), sz(1)*sz(2)*p] );
%        tmp = ipermute( reshape( expB{3,i}*tmp, [sz(3), sz(1), sz(2), p] ), [3 1 2 4] );
%
%        res = res + tmp;
%    end
%    res = reshape( res, [prod(sz), p] );
%    
%end

function res = solve_inner( L0, X, Fi, idx )
    n = size(L0, 1);
    rl = X.rank(idx);
    rr = X.rank(idx+1);

    B1 = zeros( rl );
    % calculate B1 part:
    for i = 1:idx-1
        % apply L to the i'th core
        tmp = X;
        tmp.U{i} = tensorprod_ttemps( tmp.U{i}, L0, 2 );
        B1 = B1 + innerprod( X, tmp, 'LR', idx-1);
    end

    % calculate B2 part:
    B2 = L0;

    B3 = zeros( rr );
    % calculate B3 part:
    for i = idx+1:X.order
        tmp = X;
        tmp.U{i} = tensorprod_ttemps( tmp.U{i}, L0, 2 );
        B3 = B3 + innerprod( X, tmp, 'RL', idx+1);
    end

    [V,E] = eig( kron( eye(rr), B1 ) + kron( B3, eye(rl) ) );
    E = diag(E);

    rhs = matricize( Fi, 2 ) * V;
    Y = zeros(size(rhs));
    for i=1:length(E)
        Y(:,i) = (B2 + E(i)*speye(n)) \ rhs(:,i);
    end
    res = tensorize( Y*V', 2, [rl, n, rr] );
end

function [B2, V, E] = prepare_precond( L0, X, idx )
    n = size(L0, 1);
    rl = X.rank(idx);
    rr = X.rank(idx+1);

    B1 = zeros( rl );
    % calculate B1 part:
    for i = 1:idx-1
        % apply L to the i'th core
        tmp = X;
        tmp.U{i} = tensorprod_ttemps( tmp.U{i}, L0, 2 );
        B1 = B1 + innerprod( X, tmp, 'LR', idx-1);
    end

    % calculate B2 part:
    B2 = L0;

    B3 = zeros( rr );
    % calculate B3 part:
    for i = idx+1:X.order
        tmp = X;
        tmp.U{i} = tensorprod_ttemps( tmp.U{i}, L0, 2 );
        B3 = B3 + innerprod( X, tmp, 'RL', idx+1);
    end

    [V,E] = eig( kron( eye(rr), B1 ) + kron( B3, eye(rl) ) );
    E = diag(E);
end

function res = apply_precond( B2, V, E, rhs, sz )
    n = size(B2, 1);
    rhs = reshape( rhs, sz );
    rhs = matricize( rhs, 2 ) * V;
    Y = zeros(size(rhs));
    for i=1:length(E)
        Y(:,i) = (B2 + E(i)*speye(n)) \ rhs(:,i);
    end
    res = tensorize( Y*V', 2, sz );
    res = res(:);
end



