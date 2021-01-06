% Riemannian approx. Newton for linear systems. For more information, we refer to the report
%
% D. Kressner, M. Steinlechner, and B. Vandereycken.
% Preconditioned low-rank Riemannian optimization for linear systems with tensor product structure.
% Technical report, July 2015. Revised February 2016. To appear in SIAM J. Sci. Comput.')

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

function [xR, residuum, gradnorm, cost, times] = RiemannLinsolve( L, F, X0, Lh, Ph, opts )
% L is the actual operator
%    needs a apply(L, X) interface but can be anything
% Lh is the operator that represents the (inexact) Euclidean Hessian
%    needs a apply(L, X) interface but can be anything
% Ph is the preconditioner for Lh; should be a TTeMPS_op_laplace operator
%
% When L is Laplace+perturbation, taking Lh and Ph both the Laplacian works
% well.

% set default opts
if ~exist( 'opts', 'var');                opts = struct();              end
if ~isfield( opts, 'maxiter');            opts.maxiter = 100;           end
if ~isfield( opts, 'tol');                opts.tol = 1e-16;             end
if ~isfield( opts, 'gradtol');            opts.gradtol = 1e-16;         end    
if ~isfield( opts, 'precond_tol');        opts.precond_tol = 1e-5;      end
if ~isfield( opts, 'precond_maxit');      opts.precond_maxit = 5;       end
if ~isfield( opts, 'safe_norm');          opts.safe_norm = false;       end
if ~isfield( opts, 'truncatedNewton');    opts.truncatedNewton = false; end
if ~isfield( opts, 'tediousPrec');        opts.tediousPrec = false;     end
if ~isfield( opts, 'stagtol');            opts.stagtol = inf;     end
    
if opts.truncatedNewton
    disp('Using truncated Newton with adaptive tolerance; not considering opts.precond_tol and opts.precond_maxit!')
    opts.precond_tol   = NaN;
    opts.precond_maxit = 100; 
end


d = X0.order;
n = X0.size;

[xL, xR, G] = gauge_matrices( X0 );

cost = zeros(opts.maxiter, 1);
residuum = zeros(opts.maxiter, 1);
gradnorm = zeros(opts.maxiter, 1);
times = zeros(opts.maxiter, 1);
normRHS = norm(F);
alpha = 0;

prev_res = inf;

t_start = tic;

for i = 1:opts.maxiter
    
    g = euclid_grad( L, xR, F );
    
    cost(i) = cost_function_res( xR, g );
    residuum(i) = norm(g, opts.safe_norm) / normRHS;
    times(i) = toc(t_start);
    
    grad = TTeMPS_tangent_orth( xL, xR, g );
    gradnorm(i) = norm( grad );
    sprintf('Iteration %i', i)
    if opts.truncatedNewton
        opts.precond_tol = min(0.5, sqrt(gradnorm(i))); % Nocedal-Wright 2nd edition, p169, Alg 7.1  
        sprintf('... current truncated Newton tolerance %g', opts.precond_tol)
        sprintf('... current rel. gradnorm %g, stopping at %g', gradnorm(i) / normRHS, opts.gradtol)
    end
    
    % test for stopping criterion on Residual
    if abs(residuum(i)) < opts.tol
        sprintf( 'Current residual: %g', residuum(i))
        gradnorm = gradnorm(1:i);
        residuum = residuum(1:i);
        cost = cost(1:i);
        times = times(1:i);
        sprintf( 'RiemannLinsolve CONVERGED after %i iterations', i )
        break
    end
    
    % test for stopping criterion on gradient
    if abs(gradnorm(i)/normRHS) < opts.gradtol
        sprintf( 'Current norm of Riem. gradient: %g', gradnorm(i))
        gradnorm = gradnorm(1:i);
        residuum = residuum(1:i);
        cost = cost(1:i);
        times = times(1:i);
        sprintf( 'RiemannLinsolve CONVERGED after %i iterations', i )
        break
    end
    
    residuum(i)/prev_res
    % test for stopping criterion on gradient
    if residuum(i) > opts.stagtol*prev_res
        sprintf( 'Current gradnorm reduction: %g', residuum(i)/prev_res )
        gradnorm = gradnorm(1:i);
        residuum = residuum(1:i);
        cost = cost(1:i);
        times = times(1:i);
        sprintf( 'RiemannLinsolve STAGNATED after %i iterations', i )
        break
    end
    
    P_grad = solvePrecond_noSaddle( Lh, Ph, grad, xL, xR, opts, G );      
   
    alpha = linesearch_linearized( L, P_grad, g );
    %alpha = max(-1,alpha)
    
    xR = tangentAdd( P_grad, alpha, true );
    [xL, G] = left_orth_with_gauge( xR );
    prev_res = residuum(i);
end

end

function res = cost_function_res( X, res )
res = 0.5*innerprod( X, res );
end

function res = euclid_grad( L, X, F )
res = apply(L, X) - F;
end

function alpha = linesearch_linearized( L, xi, g )
eta = tangent_to_TTeMPS( xi );
alpha = -innerprod( eta, g );
alpha = alpha / innerprod( eta, apply(L, eta) );
end
