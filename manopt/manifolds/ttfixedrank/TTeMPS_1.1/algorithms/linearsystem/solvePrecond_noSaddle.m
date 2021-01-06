%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

function xi = solvePrecond_noSaddle(L, P, rhs, xL, xR, opts, G  )
% L is the operator that represents the (inexact) Euclidean Hessian
%    needs a apply(L, X) interface but can be anything
% P is the preconditioner for Lh; should be a TTeMPS_op_laplace operator

if isa(L,'parameterdependent')
    if opts.tediousPrec
        n = xL.size;
        P_Ls = repmat( {sparse(n(2),n(2))}, 1, xR.order );
        P_Ls{1} = L.A{1};
        [P_rhs, B1, B3] = precond_laplace_overlapJacobi( P_Ls, rhs, xL, xR, G );
    else
        P_rhs = precond_rankOne( L, rhs, xL, xR );
    end
else
    % TODO make these not all the same?
    P_Ls = repmat( {P.L0}, 1, xR.order );
    % we start with the preconditioned residual, so that is one step of the
    % pcg solver if we would have started with zero initial guess
    [P_rhs, B1, B3] = precond_laplace_overlapJacobi( P_Ls, rhs, xL, xR, G );                
end

            
function y = fun_A(x)        
    % x is a vectorized tangent
    x_tangent = fill_with_vectorized( dummy, x );        
    x_ttemps = tangent_to_TTeMPS( x_tangent );
    x_ttemps = apply(L, x_ttemps);
    y_tangent = TTeMPS_tangent_orth( xL, xR, x_ttemps );
    y = vectorize_tangent( y_tangent );
end

function y = fun_P(x)        
    x_tangent = fill_with_vectorized( dummy, x );        
    if isa(L,'parameterdependent') && ~opts.tediousPrec
        eta = precond_rankOne( L, x_tangent, xL, xR );
    else
       eta = precond_laplace_overlapJacobi( P_Ls, x_tangent, xL, xR, G, B1, B3 ); 
    end
    y = vectorize_tangent( eta );
end

if opts.precond_maxit > 1
    dummy = rhs;
        
    tol = opts.precond_tol;
    maxit = opts.precond_maxit-1;  % minus one !!

    [xi_vec,flag,relres,iter] = pcg(@fun_A, vectorize_tangent(rhs), tol, maxit, @fun_P, [], vectorize_tangent(P_rhs) );
    disp(['pcg converged after ' num2str(iter) 'iterations']);
    xi = fill_with_vectorized( dummy, xi_vec );
else
    xi = P_rhs;
end

end
