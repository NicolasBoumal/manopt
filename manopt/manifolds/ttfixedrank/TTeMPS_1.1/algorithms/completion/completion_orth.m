% RTTC: Riemannian Tensor Train Completion
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
function [xL,cost,test,stats] = completion_orth( A_Omega, Omega, A_Gamma, Gamma, X, opts )
	
    if ~isfield( opts, 'maxiter');  opts.maxiter = 100;     end
    if ~isfield( opts, 'cg');       opts.cg = true;         end
    if ~isfield( opts, 'tol');      opts.tol = 1e-6;        end
    if ~isfield( opts, 'reltol');   opts.reltol = 1e-6;     end
    if ~isfield( opts, 'gradtol');  opts.gradtol = 10*eps;  end
    if ~isfield( opts, 'verbose');  opts.verbose = false;   end
	

    n = X.size;
	r = X.rank;
	
	xL = X;
	xR = orthogonalize(X, 1);

    norm_A_Omega = norm( A_Omega );
    norm_A_Gamma = norm( A_Gamma );
	
	cost = zeros(opts.maxiter,1);
	test = zeros(opts.maxiter,1);

    t = tic;
    stats.time = [];
    stats.conv = false;

	for i = 1:opts.maxiter
		grad = euclidgrad(A_Omega, xL, Omega);
		xi = TTeMPS_tangent_orth(xL, xR, grad, Omega);
		ip_xi_xi = innerprod(xi, xi);

        if sqrt( abs(ip_xi_xi) ) < opts.gradtol 
            if cost(i) < opts.tol
                disp(sprintf('CONVERGED AFTER %i STEPS. Gradient is smaller than %0.3g', ...
                      i, opts.gradtol))
                stats.conv = true;
            else
                disp('No more progress in gradient change, but not converged. Aborting!')
                stats.conv = false;
            end
            cost = cost(1:i,1);
            test = test(1:i,1);
            stats.time = [stats.time toc(t)];
            return
        end

		if (i == 1) || (~opts.cg) 
			eta = -xi;
		else
			ip_xitrans_xi = innerprod( xi_trans, xi );
			theta = ip_xitrans_xi / ip_xi_xi;
			if theta >= 0.1
                if opts.verbose, disp('steepest descent step'), end
				eta = -xi;
			else
                if opts.verbose, disp('CG step'), end
				beta = ip_xi_xi/ip_xi_xi_old;
				eta = -xi + beta*TTeMPS_tangent_orth( xL, xR, eta );
			end
		end
		
		%line search
		eta_Omega = at_Omega( eta, Omega );
		alpha = -(eta_Omega'*grad) / norm(eta_Omega)^2;
		
		X = tangentAdd( eta, alpha, true );
		xL = orthogonalize( X, X.order );
		xR = orthogonalize( X, 1 );
		cost(i) = sqrt(2*func(A_Omega, xL, Omega )) / norm_A_Omega;
		test(i) = sqrt(2*func(A_Gamma, xL, Gamma )) / norm_A_Gamma;

        if cost(i) < opts.tol
            disp(sprintf('CONVERGED AFTER %i STEPS. Rel. residual smaller than %0.3g', ...
                          i, opts.tol))
            stats.conv = true;
            cost = cost(1:i,1);
            test = test(1:i,1);
            stats.time = [stats.time toc(t)];
            return
        end

        if i > 1
            reltol = abs(cost(i) - cost(i-1)) / cost(i);
            if reltol < opts.reltol
                if cost(i) < opts.tol
                    disp(sprintf('CONVERGED AFTER %i STEPS. Relative change is smaller than %0.3g', ...
                              i, opts.reltol))
                    stats.conv = true;
                else
                    disp('No more progress in relative change, but not converged. Aborting!')
                    stats.conv = false;
                end

                cost = cost(1:i,1);
                test = test(1:i,1);
                stats.time = [stats.time toc(t)];
                return
            end
        end

		ip_xi_xi_old = ip_xi_xi;
		xi_trans = TTeMPS_tangent_orth( xL, xR, xi );

        stats.time = [stats.time toc(t)];
	end

    

end


function res = func(A_Omega, X, Omega)
	res = 0.5*norm( A_Omega - X(Omega) )^2;
end

function res = euclidgrad(A_Omega, X, Omega)
	res = X(Omega) - A_Omega;
end
