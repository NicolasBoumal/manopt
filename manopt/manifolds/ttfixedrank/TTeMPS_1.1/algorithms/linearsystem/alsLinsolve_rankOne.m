%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt
function [X, residuum, cost, times] = alsLinsolve_rankOne( L, F, X, opts )

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
            Li = contract( X, apply(L,X), idx );
            Ui = Li \ Fi(:);
            X.U{idx} = reshape( Ui, sz );

        elseif strcmpi( opts.solver, 'pcg' )

            [left, right] = Afun_prepare( L, X, idx );
            B1 =  prepare_precond( L.A{1}, X, idx );

            Ui = pcg( @(y) Afun( L, y, idx, sz, left, right), ...
                     Fi(:), ...
                     1e-10, 1000, ...
                     @(y) apply_precond( L.A{1}, B1, y, sz ), [],...
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
            Li = contract( X, apply(L, X), idx );
            Ui = Li \ Fi(:);
            X.U{idx} = reshape( Ui, sz );

        elseif strcmpi( opts.solver, 'pcg' )

            [left, right] = Afun_prepare( L, X, idx );
            B1 =  prepare_precond( L.A{1}, X, idx );

            Ui = pcg( @(y) Afun( L, y, idx, sz, left, right), ...
                     Fi(:), ...
                     1e-10, 1000, ...
                     @(y) apply_precond( L.A{1}, B1, y, sz ), [],...
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



