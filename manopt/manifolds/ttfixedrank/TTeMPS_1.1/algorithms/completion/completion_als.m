% ALS Completion
% as described in 
%   
%   Michael Steinlechner, Riemannian optimization for high-dimensional tensor completion,
%   Technical report, March 2015, revised December 2015. 
%   To appear in SIAM J. Sci. Comput. 
%

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt
function [X,cost,test,stats] = completion_als( A_Omega, Omega, A_Gamma, Gamma, X, opts )
	
    if ~isfield( opts, 'maxiter');  opts.maxiter = 100;     end
    if ~isfield( opts, 'tol');      opts.tol = 1e-6;        end
    if ~isfield( opts, 'reltol');   opts.reltol = 1e-6;     end

	n = X.size;
	r = X.rank;
    d = X.order;
	
	cost = zeros(2*opts.maxiter,1);
	test = zeros(2*opts.maxiter,1);

    norm_A_Omega = norm( A_Omega );
    norm_A_Gamma = norm( A_Gamma );

    X = orthogonalize( X, 1 );

    t = tic;
    stats.time = [0];
    stats.conv = false;

	for i = 1:opts.maxiter
        
        % ===================
        % FORWARD SWEEP:
        % ===================
        fprintf(1,'Currently optimizing core: ')
        for mu = 1:d-1
            fprintf(1,'%i ', mu)
            X.U{mu} = solve_least_squares( A_Omega, Omega, X, mu );
            X = orth_at( X, mu, 'left' );
        end
		cost(2*i-1) = sqrt(2*func(A_Omega, X, Omega)) / norm_A_Omega;
		

        if cost(2*i-1) < opts.tol 
            disp(sprintf('CONVERGED AFTER %i HALF-SWEEPS. Rel. residual smaller than %0.3g', ...
                          2*i-1, opts.tol))
            stats.conv = true;
            cost = cost(1:2*i-1,1);
            stats.time = [stats.time stats.time(end)+toc(t)];
            test(2*i-1) = sqrt(2*func(A_Gamma, X, Gamma)) / norm_A_Gamma;
            test = test(1:2*i-1,1);
            break
        end

        if i > 1
            reltol = abs(cost(2*i-1) - cost(2*i-2)) / cost(2*i-1);
            if reltol < opts.reltol
                disp(sprintf('No more progress in gradient change, but not converged after %i half-sweeps. ABORTING!. \nRelative change is smaller than %0.3g', ...
                              i, opts.reltol))
                stats.conv = false;
                cost = cost(1:2*i-1,1);
                stats.time = [stats.time stats.time(end)+toc(t)];
                test(2*i-1) = sqrt(2*func(A_Gamma, X, Gamma)) / norm_A_Gamma;
                test = test(1:2*i-1,1);
                break
            end
        end

        stats.time = [stats.time stats.time(end)+toc(t)];
        test(2*i-1) = sqrt(2*func(A_Gamma, X, Gamma)) / norm_A_Gamma;
        t = tic;

        fprintf(1,'\nFinished forward sweep.\n    Cost: %e\n    Test: %e\n', cost(2*i-1), test(2*i-1) );
        % ===================
        % BACKWARD SWEEP:
        % ===================
        fprintf(1,'Currently optimizing core: ')
        for mu = d:-1:2
            fprintf(1,'%i ', mu)
            X.U{mu} = solve_least_squares( A_Omega, Omega, X, mu );
            X = orth_at( X, mu, 'right' );
        end

		cost(2*i) = sqrt(2*func(A_Omega, X, Omega)) / norm_A_Omega;
		

        if cost(2*i) < opts.tol
            disp(sprintf('CONVERGED AFTER %i HALF-SWEEPS. Rel. residual smaller than %0.3g', ...
                          2*i, opts.tol))
            stats.conv = true;
            cost = cost(1:2*i,1);
            stats.time = [stats.time stats.time(end)+toc(t)];
            test(2*i) = sqrt(2*func(A_Gamma, X, Gamma)) / norm_A_Gamma;
            test = test(1:2*i,1);
            break
        end
        
        if i > 1
            reltol = abs(cost(2*i) - cost(2*i-1)) / cost(2*i);
            if reltol < opts.reltol
                disp(sprintf('No more progress in gradient change, but not converged after %i half-sweeps. ABORTING!. \nRelative change is smaller than %0.3g', ...
                              2*i, opts.reltol))
                stats.conv = false;
                cost = cost(1:2*i,1);
                stats.time = [stats.time stats.time(end)+toc(t)];
                test(2*i) = sqrt(2*func(A_Gamma, X, Gamma)) / norm_A_Gamma;
                test = test(1:2*i,1);
                break
            end
        end

        stats.time = [stats.time stats.time(end)+toc(t)];
        test(2*i) = sqrt(2*func(A_Gamma, X, Gamma)) / norm_A_Gamma;
        t = tic;
        fprintf(1,'\nFinished backward sweep.\n    Cost: %e\n    Test: %e\n', cost(2*i), test(2*i) );
        
        
        disp('_______________________________________________________________')
    end

    % This is to match original shape of stats.time, since we artificially start w/ [0]
    % for consistency in how we count time
    stats.time = stats.time(2:end);

end


function res = func(A_Omega, X, Omega)
	res = 0.5*norm( A_Omega - X(Omega) )^2;
end


function res = solve_least_squares( A_Omega, Omega, X, mu )

    n = X.size;
    d = X.order;
    r = X.rank;
    
    [jmu,idx] = sort(Omega(:,mu),'ascend');
    Omega = Omega(idx,:);
    A_Omega = A_Omega(idx);
    
    C = cell(1,d);
    for i=1:d
        C{i} = permute( X.U{i}, [1 3 2]);
    end
    res = zeros( size(C{mu}) );

    %B = zeros(size(Omega,1), r(mu)*r(mu+1));

    %imu = 1;
    %for sample = 1:size(Omega,1)

    %    L = 1;		
    %    for i = 1:mu-1
    %        L = L * C{i}(:,:,Omega(sample,i));
    %    end

    %    R = 1;
    %    for i = d:-1:mu+1
    %        R = C{i}(:,:,Omega(sample,i)) * R;
    %    end
    %    
    %    %B(sample,:) = kron(R',L);
    %    B(sample,:) = reshape( L'*R', 1, r(mu)*r(mu+1) );
    %end

    B = als_solve_mex( n, r, C, Omega', mu)';

    for i = 1:X.size(mu)
        idx = find(jmu == i);

        if isempty(idx) 
            error('No samples for this slice!')
        end
        res(:,:,i) = reshape(B(idx,:)\A_Omega(idx), r(mu), r(mu+1));
    end
      
   
    res = permute( res, [1 3 2] );

end
