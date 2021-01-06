% Completion for Tensor train but without individual orthogonalization 
% (TTeMPS_tangent instead of TTeMPS_tangent_orth)
%
% WARNING: use completion_orth instead!

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt
function [X,cost,test] = completion( A_Omega, Omega, A_Gamma, Gamma, X, maxiter, cg )
	
	n = X.size;
	r = X.rank;
	
	cost = zeros(maxiter,1);
	test = zeros(maxiter,1);
	for i = 1:maxiter
		grad = euclidgrad(A_Omega, X, Omega);
		xi = TTeMPS_tangent(X, grad, Omega);
		%xi_full = full(xi, X);
		ip_xi_xi = innerprod(xi, xi, X);
		if (i == 1) || (~cg) 
			eta = -xi;
		else
			ip_xitrans_xi = innerprod( xi_trans, xi, X);
			theta = ip_xitrans_xi / ip_xi_xi;
			if theta >= 0.3
				eta = -xi;
				disp('steepest descent step')
			else
				disp('CG step')
				beta = ip_xi_xi/ip_xi_xi_old;
				eta = -xi + beta*TTeMPS_tangent( X, eta );
			end
		end
		
		%line search
		eta_Omega = at_Omega( eta, Omega, X);
		alpha = -(eta_Omega'*grad) / norm(eta_Omega)^2;
		
		X = tangentAdd( eta, alpha, X, true);
		X = orthogonalize( X, X.order);
		cost(i) = func(A_Omega, X, Omega);
		test(i) = func(A_Gamma, X, Gamma);
		ip_xi_xi_old = ip_xi_xi;
		xi_trans = TTeMPS_tangent( X, xi);
	end

end


function res = func(A_Omega, X, Omega)
	res = 0.5*norm( A_Omega - X(Omega) )^2;
end

function res = euclidgrad(A_Omega, X, Omega)
	res = X(Omega) - A_Omega;
end
