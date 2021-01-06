function res = norm( x )
    %NORM Norm of a TT/MPS block-mu tensor.
    %   norm(X) computes the Frobenius norm of the TTeMPS_block tensor X.
    %   for each of the individual vectors 1 ... X.p seperately.
    %   Result is a column vector of norms with length X.p 
    %
    %   See also INNERPROD
    
    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    x = orthogonalize( x );
    block = x.U{x.mu};

    res = zeros(x.p,1);
    for i = 1:x.p
        tmp = block(:,:,:,i);
        res(i) = norm( tmp(:) );
    end

end
