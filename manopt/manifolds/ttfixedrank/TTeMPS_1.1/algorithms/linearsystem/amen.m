%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

function [X, residual, cost] = amen( L, F, X, opts )

% set default opts
if ~exist( 'opts', 'var');        opts = struct();     end
if ~isfield( opts, 'nSweeps');    opts.nSweeps = 4;    end
if ~isfield( opts, 'maxrank');    opts.maxrank = 20;   end
if ~isfield( opts, 'maxrankRes'); opts.maxrankRes = 4; end
if ~isfield( opts, 'tolRes');     opts.tolRes = 1e-13;  end
if ~isfield( opts, 'tol');        opts.tol = 1e-13;     end

d = X.order;
n = X.size;


normF = norm(F);
cost = cost_function( L, X, F );
residual = norm( apply(L, X) - F ) / normF;

for sweep = 1:opts.nSweeps
    X = orthogonalize(X, 1);
    for mu = 1:d-1
        disp( ['Current core: ', num2str(mu)] )

        % STEP 1: Solve mu-th core opimization 
        L_mu = contract( L, X, mu );
        F_mu = contract( X, F, mu );
        
        U_mu = L_mu \ F_mu(:);
        X.U{mu} = reshape( U_mu, size(X.U{mu}) );
        
        % STEP 2: Calculate current residual and cost function 
        res =  F - apply(L, X);
        residual = [residual; norm( res ) / normF];
        cost = [cost; cost_function( L, X, F )];
        disp(['Rel. residual: ' num2str(residual(end)) ', Current rank: ' num2str(X.rank) ]);

        % STEP 3: Augment mu-th and (mu+1)-th core with (truncated) residual
        R = contract( X, res, [mu, mu+1] );
        R_combined = unfold(R{1},'left') * unfold(R{2},'right'); 

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
        %X.U{mu+1} = tensorprod( right, S(1:t,1:t)*V(:,1:t)', 1);
        X.U{mu+1} = rand( t, n(mu+1), X.rank(mu+2));

    end
    for mu = d:-1:2
        disp( ['Current core: ', num2str(mu)] )

        % STEP 1: Solve mu-th core opimization 
        L_mu = contract( L, X, mu );
        F_mu = contract( X, F, mu );
        
        U_mu = L_mu \ F_mu(:);
        X.U{mu} = reshape( U_mu, size(X.U{mu}) );
        
        % STEP 2: Calculate current residual and cost function 
        res =  F - apply(L, X);
        residual = [residual; norm( res ) / normF];
        disp(['Rel. residual: ' num2str(residual(end)) ', Current rank: ' num2str(X.rank) ]);
        cost = [cost; cost_function( L, X, F )];

        % STEP 3: Augment mu-th and (mu+1)-th core with (truncated) residual
        R = contract( X, res, [mu-1, mu] );
        R_combined = unfold(R{1},'left') * unfold(R{2},'right'); 

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
        %X.U{mu+1} = tensorprod( right, S(1:t,1:t)*V(:,1:t)', 1);
        X.U{mu-1} = rand( X.rank(mu-1), n(mu-1), t);

        %residuum = [residuum; norm( apply(L, X) - F ) / normF];

        %L_mu = contract( L, X, mu );
        %F_mu = contract( X, F, mu );

        %U_mu = L_mu \ F_mu(:);
        %X.U{mu} = reshape( U_mu, size(X.U{mu}) );
        %X = orth_at( X, mu, 'right', true );
        %residuum = [residuum; norm( apply(L, X) - F ) / normF];
        %cost = [cost; cost_function( L, X, F )];
        %disp(['Rel. residual: ' num2str(residuum(end)) ', Current rank: ' num2str(X.rank) ]);
    end

end


end

function res = cost_function( L, X, F )
res = 0.5*innerprod( X, apply(L, X) ) - innerprod( X, F );
end

function res = euclid_grad( L, X, F )
res = apply(L, X) - F;
end

