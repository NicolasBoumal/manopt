function [eta inner_it stop_tCG storedb] ...
                      = tCG(problem, x, grad, eta, Delta, options, storedb)
% tCG - Truncated (Steihaug-Toint) Conjugate-Gradient
% minimize <eta,grad> + .5*<eta,Hess(eta)>
% subject to <eta,eta> <= Delta^2

% This file is part of Manopt: www.manopt.org.
% This code is an adaptation to Manopt of the original GenRTR code:
% RTR - Riemannian Trust-Region
% (c) 2004-2007, P.-A. Absil, C. G. Baker, K. A. Gallivan
% Florida State University
% School of Computational Science
% (http://www.math.fsu.edu/~cbaker/GenRTR/?page=download)
% See accompanying license file.
% The adaptation was executed by Nicolas Boumal.
% Change log: 


   % all terms involving the trust-region radius will utilize an inner product
   % w.r.t. the preconditioner; this is because the iterates grow in
   % length w.r.t. the preconditioner, guaranteeing that we will not 
   % re-enter the trust-region
   % 
   % the following recurrences for Prec-based norms and inner 
   % products come from CGT2000, pg. 205, first edition
   % below, P is the preconditioner
   % 
   % <eta_k,P*delta_k> = beta_k-1 * ( <eta_k-1,P*delta_k-1> + alpha_k-1 |delta_k-1|^2_P )
   % |delta_k|^2_P = <r_k,z_k> + beta_k-1^2 |delta_k-1|^2_P
   % 
   % therefore, we need to keep track of 
   % 1)   |delta_k|^2_P 
   % 2)   <eta_k,P*delta_k> = <eta_k,delta_k>_P
   % 3)   |eta_k  |^2_P
   % 
   % initial values are given by:
   %    |delta_0|_P = <r,z>
   %    |eta_0|_P   = 0
   %    <eta_0,delta_0>_P = 0
   % because we take eta_0 = 0
   
   import manopt.privatetools.*;
   
   theta = options.theta;
   kappa = options.kappa;

   if options.useRand, % and therefore, no preconditioner
      % eta (presumably) ~= 0 was provided by the caller   
      [Heta storedb] = getHessian(problem, x, eta, storedb);
      r = problem.M.lincomb(x, 1, grad, 1, Heta);
      e_Pe = problem.M.inner(x, eta, eta);
   else % and therefore, eta == 0
      % eta = 0*grad;
      r = grad;
      e_Pe = 0;
   end
   r_r = problem.M.inner(x, r, r);
   norm_r = sqrt(r_r);
   norm_r0 = norm_r;

   % precondition the residual
   [z storedb] = getPrecon(problem, x, r, storedb);
   
   % compute z'*r
   z_r = problem.M.inner(x, z, r);
   d_Pd = z_r;

   % initial search direction
   delta  = problem.M.lincomb(x, -1, z);
   if options.useRand, % and therefore, no preconditioner
      e_Pd = problem.M.inner(x, eta, delta);
   else % and therefore, eta == 0
      e_Pd = 0;
   end

   % pre-assume termination b/c j == end
   stop_tCG = 5;

   % begin inner/tCG loop
   j = 0;
   for j = 1 : options.maxinner,

      [Hxd storedb] = getHessian(problem, x, delta, storedb);

      % compute curvature
      d_Hd = problem.M.inner(x, delta, Hxd);

      % DEBUGGING: check that <d,Hd> = <Hd,d>
      if options.debug > 1,
         Hd_d = problem.M.inner(x, Hxd, delta);
         fprintf('DBG: |d_Hd - Hd_d| (abs/rel): %e/%e\n', abs(d_Hd-Hd_d), abs((d_Hd-Hd_d)/d_Hd));
      end

      alpha = z_r/d_Hd;
      % <neweta,neweta>_P = <eta,eta>_P + 2*alpha*<eta,delta>_P + alpha*alpha*<delta,delta>_P
      e_Pe_new = e_Pe + 2.0*alpha*e_Pd + alpha*alpha*d_Pd;

      if options.debug > 2,
         fprintf('DBG:   (r,r)  : %e\n',r_r);
         fprintf('DBG:   (d,Hd) : %e\n',d_Hd);
         fprintf('DBG:   alpha  : %e\n',alpha);
      end

      % check curvature and trust-region radius
      if d_Hd <= 0 || e_Pe_new >= Delta^2,
         % want
         %  ee = <eta,eta>_prec,x
         %  ed = <eta,delta>_prec,x
         %  dd = <delta,delta>_prec,x
         tau = (-e_Pd + sqrt(e_Pd*e_Pd + d_Pd*(Delta^2-e_Pe))) / d_Pd;
         if options.debug > 2,
            fprintf('DBG:     tau  : %e\n', tau);
         end
         eta = problem.M.lincomb(x, 1, eta, tau, delta);
         if d_Hd <= 0,
            stop_tCG = 1;     % negative curvature
         else
            stop_tCG = 2;     % exceeded trust region
         end
         break;
      end

      % no negative curvature and eta_prop inside TR: accept it
      e_Pe = e_Pe_new;
      eta = problem.M.lincomb(x, 1, eta, alpha, delta);

      % update the residual
      r = problem.M.lincomb(x, 1, r, alpha, Hxd);
      % re-tangentalize r
	  % NB Feb. 12, 2013 : this is not necessary, and does not correspond to the notion of projector
	  % in Manopt anymore (as of version 1.0.1)
      % r = problem.M.proj(x, r);

      % compute new norm of r
      r_r = problem.M.inner(x, r, r);
      norm_r = sqrt(r_r);

      % check kappa/theta stopping criterion
      if j >= options.mininner && norm_r <= norm_r0*min(norm_r0^theta, kappa)
         % residual is small enough to quit
         if kappa < norm_r0^theta,
             stop_tCG = 3;  % linear convergence
         else
             stop_tCG = 4;  % superlinear convergence
         end
         break;
      end

      % precondition the residual
     [z storedb] = getPrecon(problem, x, r, storedb);

      % save the old z'*r
      zold_rold = z_r;
      % compute new z'*r
      z_r = problem.M.inner(x, z, r);

      % compute new search direction
      beta = z_r/zold_rold;
      delta = problem.M.lincomb(x, -1, z, beta, delta);
      
      % update new P-norms and P-dots
      e_Pd = beta*(e_Pd + alpha*d_Pd);
      d_Pd = z_r + beta*beta*d_Pd;

   end  % of tCG loop
   inner_it = j;

end
