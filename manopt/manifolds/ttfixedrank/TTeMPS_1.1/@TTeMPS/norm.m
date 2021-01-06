function res = norm( x, safe )
    %NORM Norm of a TT/MPS tensor.
    %   norm(X) computes the Frobenius norm of the TT/MPS tensor X.
    %
    %   norm(X, SAFE) with SAFE=true computes the Frobenius norm of the TT/MPS tensor X
    %       with reorthogonalization to increase the accuracy
    %
    %   See also INNERPROD
    
    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt
    
    if ~exist('safe','var')
        safe = true;
    end
    
    if safe
        x = orthogonalize(x, x.order );
        res = norm( x.U{end}(:) );
    else
        res = sqrt(innerprod( x, x )); 

        if res < 1e-7
            x = orthogonalize(x, x.order );
            res = norm( x.U{end}(:) );
        end
    end
    
end
