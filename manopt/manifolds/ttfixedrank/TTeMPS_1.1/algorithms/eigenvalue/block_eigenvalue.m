function [X, C, evalue, residuums, micro_res, objective, elapsed_time] = block_eigenvalue(A, p, rr, opts)
    %BLOCK_EIGENVALUE Calculate p smallest eigenvalues of a TTeMPS operator
    %
    %   [X, C, evalue, residuums, micro_res, objective, elapsed_time] = block_eigenvalue(A, P, RR, OPTS)
    %       performs a block-eigenvalue optimization scheme to compute the p smallest eigenvalues of A
    %       using the algorithm described in [1]. 
    %
    %   RR defines the starting rank and should usually be set to ones(1,d) where d is the dimensionality.
    %   If p == 1, the algorithm equals a standard ALS-procedure for the eigenvalue problem, which is NOT 
    %   rank adaptive. Hence, in this case RR should be taken to be the expected rank of the solution or, 
    %   if unknown, the highest affordable rank.
    %   
    %   Specify the wanted options using the OPTS argument. All options have
    %   default values (denoted in brackets). Hence, you only have to specify those you want 
    %   to change.
    %   The OPTS struct is organized as follows: 
	%       OPTS.maxiter        Maximum number of full sweeps to perform        [3]
	%       OPTS.maxrank        Maximum rank during the iteration               [40]
	%       OPTS.tol            Tolerance for the shift from one core 
    %                           to the next                                     [1e-8]
	%       OPTS.tolLOBPCG      Tolerance for inner LOBPCG solver               [1e-6]
	%       OPTS.maxiterLOBPCG  Max. number of iterations for LOBPCG            [2000]
	%       OPTS.verbose        Show iteration information (true/false)         [true]
	%       OPTS.precInner      Precondition the LOBPCG (STRONGLY RECOMMENDED!) [true]
    %
    %
    %   NOTE: To run block_eigenvalue, Knyazev's implementation of LOBPCG is required. The corresponding
    %         m-file can be downloaded from 
    %               http://www.mathworks.com/matlabcentral/fileexchange/48-lobpcg-m
    %
	%	References: 
	%	[1] S.V. Dolgov, B.N. Khoromskij, I.V. Oseledets, D.V. Savostyanov
    %		Computation of extreme eigenvalues in higher dimensions using block tensor train format
    %	    Computer Physics Communications 185 (2014) 1207-1216.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2014
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt


nn = A.size_col;
d = A.order;

if ~isfield( opts, 'maxiter');       opts.maxiter = 3;           end
if ~isfield( opts, 'maxrank');       opts.maxrank = 40;          end
if ~isfield( opts, 'tol');           opts.tol = 1e-8;            end
if ~isfield( opts, 'tolLOBPCG');     opts.tolLOBPCG = 1e-6;      end
if ~isfield( opts, 'maxiterLOBPCG'); opts.maxiterLOBPCG = 2000;  end
if ~isfield( opts, 'verbose');       opts.verbose = 1;           end
if ~isfield( opts, 'precInner');     opts.precInner = true;      end
if ~isfield( opts, 'X0');            useX0 = 0; else useX0 = 1;  end

if p == 1
    tolLOBPCGmod = opts.tolLOBPCG;
else
    tolLOBPCGmod = 0.00001;
end


if ~useX0
    X = TTeMPS_rand( rr, nn ); 
    % Left-orthogonalize the tensor:
    X = orthogonalize( X, X.order );

    X.U{d} = rand( X.rank(d), X.size(d), X.rank(d+1), p );
    C = cell( 1, p );
    C{1} = X.U{d}(:,:,:,1);
    for i = 2:p
        C{i} = X.U{d}(:,:,:,i);
    end

else
    X = opts.X0;
    X.U{d} = rand( X.rank(d), X.size(d), X.rank(d+1), p );
    C = cell( 1, p );
    C{1} = X.U{d}(:,:,:,1);
    for i = 2:p
        C{i} = X.U{d}(:,:,:,i);
    end;
end

evalue = [];
residuums = zeros(p,2*opts.maxiter);
micro_res = [];
resi_norm = zeros(p,1);
objective = [];
tic;
elapsed_time = [];


for i = 1:opts.maxiter
    
    disp(sprintf('Iteration %i:', i));
    % right-to-left sweep
    fprintf(1, 'RIGHT-TO-LEFT SWEEP. ---------------\n') 

    for mu = d:-1:2
        sz = [X.rank(mu), X.size(mu), X.rank(mu+1)];
        % shows itertaion information
        if opts.verbose
            fprintf(1,'Current core: %i. Current iterate (first eigenvalue): \n', mu);
            disp(X);
            fprintf(1,'Running LOBPCG: system size %i, tol %g ... ', prod(sz), max(opts.tolLOBPCG, min( tolLOBPCGmod, sqrt(sum(residuums(:,2*(i-1)+1).^2))/sqrt(p)*tolLOBPCGmod )))
        else
            fprintf(1,'%i  ',mu);
        end
        
        [left, right] = Afun_prepare( A, X, mu );

		if opts.precInner
            if prod(sz) < 5*p;
                Amat = zeros( prod(sz) );
                for i_mat = 1:prod(sz)
                    Amat(:,i_mat) = Afun_block_optim( A, [zeros(i_mat-1,1); 1; zeros(prod(sz)-i_mat,1)], sz, left, right, mu);
                end
                [vmat,emat] = eig(Amat);
                [emat,sortind] = sort(diag(emat),'ascend');
                vmat = vmat(:, sortind);
                L = emat(1:p);
                V = vmat(:, 1:p);
                
                disp('Reduced system too small for LOBPCG, using eig instead');
                disp(['Current Eigenvalue approx: ', num2str( L(1:p)')]);

                
            else
                expB = constr_precond_inner( A, X, mu );
                [V,L,failureFlag,Lhist ] = lobpcg( rand( prod(sz), p), ...
                                                   @(y) Afun_block_optim( A, y, sz, left, right, mu), [], ...
                                                   @(y) apply_local_precond( A, y, sz, expB), ...
                                                   max(opts.tolLOBPCG, min( tolLOBPCGmod, sqrt(sum(residuums(:,2*(i-1)+1).^2))/sqrt(p)*tolLOBPCGmod )), ...
                                                   opts.maxiterLOBPCG, 0);
                if opts.verbose
                    if failureFlag
                        fprintf(1,'NOT CONVERGED within %i steps!\n', opts.maxiterLOBPCG)
                    else
                        fprintf(1,'converged after %i steps!\n', size(Lhist,2));
                    end
                    disp(['Current Eigenvalue approx: ', num2str( L(1:p)')]);
                end
            end
        else
            if prod(sz) < 25;
                Amat = zeros( prod(sz) );
                for i_mat = 1:prod(sz)
                    Amat(:,i_mat) = Afun_block_optim( A, [zeros(i_mat-1,1); 1; zeros(prod(sz)-i_mat,1)], sz, left, right, mu);
                end
                [vmat,emat] = eig(Amat);
                [emat,sortind] = sort(diag(emat),'ascend');
                vmat = vmat(:, sortind);
                L = emat(1:p);
                V = vmat(:, 1:p);
                
                disp('Condition number!')
                cond(Amat)
                
                disp('Reduced system too small for LOBPCG, using eig instead');
                disp(['Current Eigenvalue approx: ', num2str( L(1:p)')]);

            else
                X0 = rand( prod(sz), p);
                X0 = orth(X0);
                [V,L,failureFlag,Lhist ] = lobpcg( X0, ...
                                               @(y) Afun_block_optim( A, y, sz, left, right, mu), ...
                                               opts.tolLOBPCG, opts.maxiterLOBPCG, 0);
            end
        end

        evalue = [evalue, L(1:p)];
        objective = [objective, sum(evalue(:,end))];
        elapsed_time = [elapsed_time, toc];

        X.U{mu} = reshape( V, [sz, p] );
        lamX = X;
        lamX.U{mu} = repmat( reshape(L, [1 1 1 p]), [X.rank(mu), X.size(mu), X.rank(mu+1), 1]).*lamX.U{mu};
        res_new = apply(A, X) - lamX

        for j = 1:p
            X.U{mu} = reshape( V(:,j), sz );
            res = apply(A, X) - L(j)*X;
            resi_norm(j) = norm(res);
            residuums(j,2*(i-1)+1) = resi_norm(j);
        
        end
        
        
        micro_res = [micro_res, resi_norm];

        % split new core
        V = reshape( V, [sz, p] );
        V = permute( V, [1, 4, 2, 3] );
        V = reshape( V, [sz(1)*p, sz(2)*sz(3)] );

        [U,S,V] = svd( V, 'econ' );
        
        if p == 1
            s = length(diag(S));
        else
            s = trunc_singular( diag(S), opts.tol, true, opts.maxrank );
        end

        V = V(:,1:s)';
        X.U{mu} = reshape( V, [s, sz(2), sz(3)] );

        W = U(:,1:s)*S(1:s,1:s);
        W = reshape( W, [sz(1), p, s]);
        W = permute( W, [1, 3, 2]);
        for k = 1:p
            C{k} = tensorprod( X.U{mu-1}, W(:,:,k)', 3);
        end
        
        X.U{mu-1} = C{1};

		if opts.verbose
	        disp( ['Augmented system of size (', num2str( [sz(1)*p, sz(2)*sz(3)]), '). Cut-off tol: ', num2str(opts.tol) ])
	        disp( sprintf( 'Number of SVs: %i. Truncated to: %i => %g %%', length(diag(S)), s, s/length(diag(S))*100))
	    	disp(' ')
		end
    end

    % calculate current residuum

    fprintf(1, '---------------    finshed sweep.    ---------------\n') 
    disp(['Current residuum: ', num2str(residuums(:,2*(i-1)+1).')])
    disp(' ')
    disp(' ')
    fprintf(1, '--------------- LEFT-TO-RIGHT SWEEP. ---------------\n') 
    % left-to-right sweep
    for mu = 1:d-1
		sz = [X.rank(mu), X.size(mu), X.rank(mu+1)];

        if opts.verbose
            fprintf(1,'Current core: %i. Current iterate (first eigenvalue): \n', mu);
            disp(X);
            fprintf(1,'Running LOBPCG: system size %i, tol %g ... ', prod(sz), max(opts.tolLOBPCG, min( tolLOBPCGmod, sqrt(sum(residuums(:,2*(i-1)+2).^2))/sqrt(p)*tolLOBPCGmod )))
        else
            fprintf(1,'%i  ',mu);
        end

        [left, right] = Afun_prepare( A, X, mu );

		if opts.precInner
            expB = constr_precond_inner( A, X, mu );
            X0 = rand( prod(sz), p);
            X0 = orth(X0);
            [U,L,failureFlag,Lhist ] = lobpcg( X0 , ...
                                               @(y) Afun_block_optim( A, y, sz, left, right, mu), [], ...
                                               @(y) apply_local_precond( A, y, sz, expB), ...
                                               max(opts.tolLOBPCG, min( tolLOBPCGmod, sqrt(sum(residuums(:,2*(i-1)+2).^2))/sqrt(p)*tolLOBPCGmod )), ...
                                               opts.maxiterLOBPCG, 0);
        else
            if prod(sz) < 25;
                Amat = zeros( prod(sz) );
                for i_mat = 1:prod(sz)
                    Amat(:,i_mat) = Afun_block_optim( A, [zeros(i_mat-1,1); 1; zeros(prod(sz)-i_mat,1)], sz, left, right, mu);
                end
                [vmat,emat] = eig(Amat);
                [emat,sortind] = sort(diag(emat),'ascend');
                vmat = vmat(:, sortind);
                L = emat(1:p);
                U = vmat(:, 1:p);
                
                disp('Reduced system too small for LOBPCG, using eig instead');
                disp(['Current Eigenvalue approx: ', num2str( L(1:p)')]);
            else
                X0 = rand( prod(sz), p);
                X0 = orth(X0);
                [U,L,failureFlag,Lhist ] = lobpcg( X0 , ...
                                                @(y) Afun_block_optim( A, y, sz, left, right, mu), ...
                                                opts.tolLOBPCG, opts.maxiterLOBPCG, 0);
            end
        end

        if opts.verbose
            if failureFlag
                fprintf(1,'NOT CONVERGED within %i steps!\n', opts.maxiterLOBPCG)
            else
                fprintf(1,'converged after %i steps!\n', size(Lhist,2));
            end
            disp(['Current Eigenvalue approx: ', num2str( L(1:p)')]);
        end
        evalue = [evalue, L(1:p)];
        objective = [objective, sum(evalue(:,end))];
        elapsed_time = [elapsed_time, toc];
    
        for j = 1:p
            X.U{mu} = reshape( U(:,j), sz );
            res = apply(A, X) - L(j)*X;
            resi_norm(j) = norm(res);
            residuums(j,2*(i-1)+2) = resi_norm(j);
        end
        micro_res = [micro_res, resi_norm];

        % split new core
        U = reshape( U, [sz, p] );
        U = permute( U, [1, 2, 4, 3] );
        U = reshape( U, [sz(1)*sz(2), p*sz(3)] );

        [U,S,V] = svd( U, 'econ' );
        if p == 1
            s = length(diag(S));
        else
            s = trunc_singular( diag(S), opts.tol, true, opts.maxrank );
        end

        U = U(:,1:s);
        X.U{mu} = reshape( U, [sz(1), sz(2), s] );
        W = S(1:s,1:s)*V(:,1:s)';
        W = reshape( W, [s, p, sz(3)]);
        W = permute( W, [1, 3, 2]);
        
        for k = 1:p
            C{k} = tensorprod( X.U{mu+1}, W(:,:,k), 1);
        end
        
        X.U{mu+1} = C{1};

		if opts.verbose
		    disp( ['Augmented system of size (', num2str( [sz(1)*sz(2), p*sz(3)]), '). Cut-off tol: ', num2str(opts.tol) ])
		    disp( sprintf( 'Number of SVs: %i. Truncated to: %i => %g %%', length(diag(S)), s, s/length(diag(S))*100))
			disp(' ')
		end
    end

    fprintf(1, '---------------    finshed sweep.    ---------------\n') 
    disp(['Current residuum: ', num2str(residuums(:,2*(i-1)+2).')])
    disp(' ')
    disp(' ')
end

evs = zeros(p,1);
% Evaluate rayleigh quotient for the p TT/MPS tensors
for i=1:p
    evec = X;
    evec.U{d} = C{i};
    evs(i) = innerprod( evec, apply(A, evec));
end
evalue = [evalue, evs];

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

function res = Afun_block_optim( A, U, sz, left, right, mu )

    p = size(U, 2);
    
    V = reshape( U, [sz, p] );
    V = A.apply( V, mu );
    
    if mu == 1
        tmp = reshape( permute( V, [3 1 2 4] ), [size(V, 3), sz(1)*sz(2)*p]);
        tmp = right * tmp;
        tmp = reshape( tmp, [size(right, 1), sz(1), sz(2), p]);
        tmp = ipermute( tmp, [3 1 2 4] );
    elseif mu == A.order
        tmp = reshape( V, [size(V,1), sz(2)*sz(3)*p]);
        tmp = left * tmp;
        tmp = reshape( tmp, [size(left, 1), sz(2), sz(3), p]);
    else
        tmp = reshape( permute( V, [3 1 2 4] ), [size(V, 3), size(V, 1)*sz(2)*p]);
        tmp = right * tmp;
        tmp = reshape( tmp, [size(right, 1), size(V, 1), sz(2), p]);
        tmp = ipermute( tmp, [3 1 2 4] );

        tmp = reshape( tmp, [size(V, 1), sz(2)*sz(3)*p]);
        tmp = left * tmp;
        tmp = reshape( tmp, [size(left, 1), sz(2), sz(3), p]);

    end

    res = reshape( tmp, [prod(sz), p] );
        

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

