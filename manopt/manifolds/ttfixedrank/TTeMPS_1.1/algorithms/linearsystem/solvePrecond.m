%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

function xi = solvePrecond(L, P, rhs, xL, xR, opts  )

if nargin==5
    opts.precond_tol = 1e-5;   % 1e-8 has the same performace, just to be sure...
    opts.precond_maxit = 5;
end

P_rhs = precond_laplace( P, rhs );


            
    function y = fun_A(x)        
        % x is a vectorized tangent
        x_tangent = fill_with_vectorized( dummy, x );        
        x_ttemps = tangent_to_TTeMPS( x_tangent );
        y_ttemps = apply(L, x_ttemps);
        y_tangent = TTeMPS_tangent_orth( xL, xR, y_ttemps );
        y = vectorize_tangent( y_tangent );
    end

    function y = fun_P(x)        
        x_tangent = fill_with_vectorized( dummy, x );        
        eta = precond_laplace( P, x_tangent );
        y = vectorize_tangent( eta );
    end

if opts.precond_maxit > 1
    dummy = rhs;
        
    tol = opts.precond_tol;
    maxit = opts.precond_maxit-1;  % minus one !!

    [xi_vec,flag,relres,iter] = pcg(@fun_A, vectorize_tangent(rhs), tol, maxit, @fun_P, [], vectorize_tangent(P_rhs) );
    iter
    xi = fill_with_vectorized( dummy, xi_vec );
else
    xi = P_rhs;
end

end
