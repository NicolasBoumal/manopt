%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

function [X, residual, cost, times] = amen_fast( L, F, X, opts )

t_start = tic();
% set default opts
if ~exist( 'opts', 'var');        opts = struct();     end
if ~isfield( opts, 'nSweeps');    opts.nSweeps = 4;    end
if ~isfield( opts, 'maxrank');    opts.maxrank = 20;   end
if ~isfield( opts, 'maxrankRes'); opts.maxrankRes = 4; end
if ~isfield( opts, 'tolRes');     opts.tolRes = 1e-13;  end
if ~isfield( opts, 'tol');        opts.tol = 1e-13;     end
if ~isfield( opts, 'solver');     opts.solver = 'direct';   end
if ~isfield( opts, 'prec');       opts.prec = true;   end
    

d = X.order;
n = X.size;

normF = norm(F);
cost = cost_function( L, X, F );
residual = norm( apply(L, X) - F ) / normF;
times = toc(t_start);

for sweep = 1:opts.nSweeps
    X = orthogonalize(X, 1);
    for mu = 1:d-1
        disp( ['Current core: ', num2str(mu)] )

        % STEP 1: Solve mu-th core opimization
        F_mu = contract( X, F, mu );
        sz = [X.rank(mu), X.size(mu), X.rank(mu+1)];
        
        if strcmpi( opts.solver, 'direct' ) 
            % if system very small
            L_mu = contract( L, X, mu );
            U_mu = L_mu \ F_mu(:);
            X.U{mu} = reshape( U_mu, sz );
        elseif strcmpi( opts.solver, 'pcg' )
            [left, right] = Afun_prepare( L, X, mu );
            [B2, V, E] =  prepare_precond( L.L0, X, mu );

            U_mu = pcg( @(y) Afun( L, y, mu, sz, left, right), ...
                     F_mu(:), ...
                     1e-10, 1000, ...
                     @(y) apply_precond( B2, V, E, y, sz ), [],...
                     X.U{mu}(:) ); 
            X.U{mu} = reshape( U_mu, sz );
        else
            error( 'Unknown opts.solver type. Use either ''direct'' (default) or ''pcg''.' )
        end
        
        % STEP 2: Calculate current residual and cost function 
        res =  F - apply(L, X);
        residual = [residual; norm( res ) / normF];
        cost = [cost; cost_function( L, X, F )];
        disp(['Rel. residual: ' num2str(residual(end)) ', Current rank: ' num2str(X.rank) ]);

        % STEP 3: Augment mu-th and (mu+1)-th core with (truncated) residual
        R = contract( X, res, [mu, mu+1] );
        R_combined = unfold(R{1},'left') * unfold(R{2},'right'); 
        if opts.prec
            R_combined = precond_residual( L.L0, X, R_combined, mu );
        end
        [uu,ss,~] = svd( R_combined, 'econ');  
        s = find( diag(ss) > opts.tolRes*norm(diag(ss)), 1, 'last' );
		if opts.maxrankRes ~= 0 
			s = min( s, opts.maxrankRes );
		end
        R{1} = reshape( uu(:,1:s)*ss(1:s,1:s), [X.rank(mu), n(mu), s]);
        %R{2} = reshape( vv(:,1:s)', [s, n(mu+1), X.rank(mu+2)]);

        left = cat(3, X.U{mu}, R{1});
        %right = cat(1, X.U{mu+1}, R{2});

        % STEP 4: Move orthogonality to (mu+1)-th core while performing rank truncation 
        [U,S,~] = svd( unfold(left,'left'), 'econ' );
        t = find( diag(S) > opts.tol*norm(diag(S)), 1, 'last' );
        t = min( t, opts.maxrank );
        X.U{mu} = reshape( U(:,1:t), [X.rank(mu), n(mu), t] );
        X.U{mu+1} = rand( t, n(mu+1), X.rank(mu+2));

        times = [times; toc(t_start)];
    end
    for mu = d:-1:2
        disp( ['Current core: ', num2str(mu)] )

        % STEP 1: Solve mu-th core opimization 
        F_mu = contract( X, F, mu );
        sz = [X.rank(mu), X.size(mu), X.rank(mu+1)];
    
        if strcmpi( opts.solver, 'direct' ) 
            L_mu = contract( L, X, mu );
            U_mu = L_mu \ F_mu(:);
            X.U{mu} = reshape( U_mu, size(X.U{mu}) );
        elseif strcmpi( opts.solver, 'pcg' )
            [left, right] = Afun_prepare( L, X, mu );
            [B2, V, E] =  prepare_precond( L.L0, X, mu );

            U_mu = pcg( @(y) Afun( L, y, mu, sz, left, right), ...
                     F_mu(:), ...
                     1e-10, 1000, ...
                     @(y) apply_precond( B2, V, E, y, sz ), [],...
                     X.U{mu}(:) ); 
            X.U{mu} = reshape( U_mu, sz );
        else
            error( 'Unknown opts.solver type. Use either ''direct'' (default) or ''diag''.' )
        end
        
        % STEP 2: Calculate current residual and cost function 
        res =  F - apply(L, X);
        residual = [residual; norm( res ) / normF];
        disp(['Rel. residual: ' num2str(residual(end)) ', Current rank: ' num2str(X.rank) ]);
        cost = [cost; cost_function( L, X, F )];

        % STEP 3: Augment mu-th and (mu+1)-th core with (truncated) residual
        R = contract( X, res, [mu-1, mu] );
        R_combined = unfold(R{1},'left') * unfold(R{2},'right'); 
        if opts.prec
            R_combined = precond_residual( L.L0, X, R_combined, mu-1 );
        end
        [~,ss,vv] = svd( R_combined, 'econ');  
        s = find( diag(ss) > opts.tolRes*norm(diag(ss)), 1, 'last' );
		if opts.maxrankRes ~= 0 
			s = min( s, opts.maxrankRes );
		end
        R{2} = reshape( ss(1:s,1:s)*vv(:,1:s)', [s, n(mu), X.rank(mu+1)]);

        right = cat(1, X.U{mu}, R{2});

        % STEP 4: Move orthogonality to (mu+1)-th core while performing rank truncation 
        [~,S,V] = svd( unfold(right,'right'), 'econ' );
        t = find( diag(S) > opts.tol*norm(diag(S)), 1, 'last' );
        t = min( t, opts.maxrank );
        X.U{mu} = reshape( V(:,1:t)', [t, n(mu), X.rank(mu+1)] );
        X.U{mu-1} = rand( X.rank(mu-1), n(mu-1), t);
        
        times = [times; toc(t_start)];
    end

end


end

function res = cost_function( L, X, F )
res = 0.5*innerprod( X, apply(L, X) ) - innerprod( X, F );
end

function res = euclid_grad( L, X, F )
res = apply(L, X) - F;
end

function res = precond_residual( L0, X, R_combined, idx )
    n = size(L0, 1);
    rl = X.rank(idx);
    rr = X.rank(idx+2);

    B1 = zeros( rl );
    % calculate B1 part:
    for i = 1:idx-1
        % apply L to the i'th core
        tmp = X;
        tmp.U{i} = tensorprod( tmp.U{i}, L0, 2 );
        B1 = B1 + innerprod( X, tmp, 'LR', idx-1);
    end

    % calculate B2 part:
    B2 = kron( L0, speye(n) ) + kron( speye(n), L0 );

    B3 = zeros( rr );
    % calculate B3 part:
    for i = idx+2:X.order
        tmp = X;
        tmp.U{i} = tensorprod( tmp.U{i}, L0, 2 );
        B3 = B3 + innerprod( X, tmp, 'RL', idx+2);
    end

    [V,E] = eig( kron( eye(rr), B1 ) + kron( B3, eye(rl) ) );
    E = diag(E);

    R_combined = reshape( R_combined, [rl, n*n, rr] );
    rhs = matricize( R_combined, 2 ) * V;
    Y = zeros(size(rhs));
    for i=1:length(E)
        Y(:,i) = (B2 + E(i)*speye(n*n)) \ rhs(:,i);
    end
    res = tensorize( Y*V', 2, [rl, n*n, rr] );
    res = reshape( res, [rl*n, n*rr] );
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
function [B2, V, E] = prepare_precond( L0, X, idx )
    n = size(L0, 1);
    rl = X.rank(idx);
    rr = X.rank(idx+1);

    B1 = zeros( rl );
    % calculate B1 part:
    for i = 1:idx-1
        % apply L to the i'th core
        tmp = X;
        tmp.U{i} = tensorprod( tmp.U{i}, L0, 2 );
        B1 = B1 + innerprod( X, tmp, 'LR', idx-1);
    end

    % calculate B2 part:
    B2 = L0;

    B3 = zeros( rr );
    % calculate B3 part:
    for i = idx+1:X.order
        tmp = X;
        tmp.U{i} = tensorprod( tmp.U{i}, L0, 2 );
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


