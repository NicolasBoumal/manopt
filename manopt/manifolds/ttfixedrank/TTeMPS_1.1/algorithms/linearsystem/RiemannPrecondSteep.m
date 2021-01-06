%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

function [xR, residuum, gradnorm, cost, times] = RiemannPrecondSteep( L, F, X0, Lh, Ph, opts )
% L is the actual operator
%    needs a apply(L, X) interface but can be anything
% Lh is the operator that represents the (inexact) Euclidean Hessian
%    needs a apply(L, X) interface but can be anything
% Ph is the preconditioner for Lh; should be a TTeMPS_op_laplace operator
%
% When L is Laplace+perturbation, taking Lh and Ph both the Laplacian works
% well.

t_start = tic();
% set default opts
if ~exist( 'opts', 'var');       opts = struct();     end
if ~isfield( opts, 'maxiter');   opts.maxiter = 500;  end
if ~isfield( opts, 'tol');       opts.tol = 1e-16;     end
if ~isfield( opts, 'safe_norm');       opts.safe_norm = false;     end

d = X0.order;
n = X0.size;



[xL, xR, G] = gauge_matrices( X0 );

%xL = orthogonalize(X, d);
%xR = orthogonalize(X, 1);

cost = zeros(opts.maxiter, 1);
residuum = zeros(opts.maxiter, 1);
gradnorm = zeros(opts.maxiter, 1);
times = zeros(opts.maxiter, 1);
normRHS = norm(F);

for i = 1:opts.maxiter
    
    g = euclid_grad( L, xR, F );
    
    cost(i) = cost_function_res( xR, g );
    residuum(i) = norm(g, opts.safe_norm) / normRHS;
    times(i) = toc(t_start);
    
    % test for stopping criterion
    if abs(residuum(i)) < opts.tol
        sprintf( 'Current residual: %g', residuum(i))
        residuum = residuum(1:i);
        cost = cost(1:i);
        times = times(1:i);
        sprintf( 'RiemannLinsolve CONVERGED after %i iterations', i )
        break
    end
    
    grad = TTeMPS_tangent_orth( xL, xR, g );
    gradnorm(i) = norm( grad );
    
    sprintf('steepest descent step %i', i)
    %P_grad = solvePrecond( L, P, grad, xL, xR, opts );
    P_grad = solvePrecond_noSaddle( Lh, Ph, grad, xL, xR, opts, G );      
        
       
        
    %check_precond_laplace(P, grad, P_grad)
    %eta = -P_grad;       
    
    %line search
    alpha = linesearch_linearized( L, P_grad, g )
    %alpha = linesearch_linearized2( L, P_grad, grad )
    %alpha = -1;
    xR = tangentAdd(  P_grad, alpha, true );
    
    [xL, G] = left_orth_with_gauge( xR );
    %xL = orthogonalize( X, d );
    %xR = orthogonalize( X, 1 );
end

end

function res = cost_function( L, X, F )
res = 0.5*innerprod( X, apply(L, X) ) - innerprod( X, F );
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

function alpha = linesearch_linearized2( L, xi, grad )
alpha = -innerprod( xi, grad );
eta = tangent_to_TTeMPS( xi );
alpha = alpha / innerprod( eta, apply(L, eta) );
end
