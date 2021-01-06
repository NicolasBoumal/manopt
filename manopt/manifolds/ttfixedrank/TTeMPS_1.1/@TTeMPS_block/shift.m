%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt
function x = shift( x, nu, tol, maxrank )
    
    r = x.rank;
    n = x.size;
    p = x.p;
    mu = x.mu;

    if ~exist('tol', 'var')
        tol = eps;
    end
    if ~exist('maxrank', 'var')
        maxrank = inf;%r(mu+1);
    end


    if mu == nu-1   
        % shift block one to the right
        U = permute( x.U{mu}, [1, 2, 4, 3] );
        U = reshape( U, [r(mu)*n(mu), p*r(mu+1)] );

        [U,S,V] = svd( U, 'econ' );
        if p == 1 
            s = length(diag(S));
        else
            s = trunc_singular( diag(S), tol );
        end
        if length(diag(S)) >= s+1
            disp(['cut singular value of rel. magnitude (s_{i+1}/s_1): ', ...
                        num2str(S(s+1,s+1)/S(1,1))])
        end
        U = U(:,1:s);
        x.U{mu} = reshape( U, [r(mu), n(mu), s] );
        W = S(1:s,1:s)*V(:,1:s)';
        W = reshape( W, [s, p, r(mu+1)]);
        W = permute( W, [1, 3, 2]);
        
        C = zeros( [s, n(nu), r(nu+1), p] ); 
        for k = 1:p
            C(:,:,:,k) = tensorprod( x.U{nu}, W(:,:,k), 1);
        end
        
        x.U{nu} = C;
        x.mu = nu;

    elseif x.mu == nu+1   
        % shift block one to the left
        V = permute( x.U{mu}, [1, 4, 2, 3] );
        V = reshape( V, [r(mu)*p, n(mu)*r(mu+1)] );

        [U,S,V] = svd( V, 'econ' );
        if p == 1
            s = length(diag(S));
        else
            s = trunc_singular( diag(S), tol );
        end
        if length(diag(S)) >= s+1
            disp(['cut singular value of rel. magnitude (s_{i+1}/s_1): ', ...
                        num2str(S(s+1,s+1)/S(1,1))])
        end
        V = V(:,1:s)';
        x.U{mu} = reshape( V, [s, n(mu), r(mu+1)] );

        W = U(:,1:s)*S(1:s,1:s);
        W = reshape( W, [r(mu), p, s]);
        W = permute( W, [1, 3, 2]);

        C = zeros( [r(nu), n(nu), s, p] ); 
        for k = 1:p
            C(:,:,:,k) = tensorprod( x.U{nu}, W(:,:,k)', 3);
        end
        
        x.U{nu} = C;
        x.mu = nu;
    else
        error('Can only shift the superblock one core left or right')
    end


end
