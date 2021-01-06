function x = from_array(A,opt)
    %FROM_ARRAY Approximate full array by TTeMPS tensor of prescribed rank 
    %   or within a prescribed tolerance.
    %
    %   X = TTeMPS.from_array( A, tol ) approximates the given array A by a
    %       TTeMPS tensor such that the the error is in the order of tol.
    %
    %   X = TTeMPS.from_array( A, r ), with r a vector of length (ndims(A))+1),
    %       approximates the given array A by a rank-r TTeMPS tensor, such that 
    %       X.rank = r.
    %       

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    n = size(A);
    d = length(n);

    if length(opt) == 1
        useTol = true;
        tol = opt;
        r = ones(1,d+1);
    else
        useTol = false;
        r = opt;
        if r(1) ~= 1 || r(d+1) ~= 1
            error('Invalid rank specified')
        end
    end

    U = cell(1,d);

    % process from left to right
    % first core
    A = reshape( A, n(1), prod(n(2:end)));
    [u,s,v] = svd(A,'econ');
    if useTol
        r(2) = trunc_singular( diag(s), tol );
    end
    U{1} = reshape( u(:,1:r(2)), [1, n(1), r(2)] );
    A = s(1:r(2),1:r(2))*v(:,1:r(2))';

    % middle cores
    for i = 2:d-1
        A = reshape( A, n(i)*r(i), prod(n(i+1:end)));
        [u,s,v] = svd(A,'econ');
        if useTol
            r(i+1) = trunc_singular( diag(s), tol );
        end
        U{i} = reshape( u(:,1:r(i+1)), [r(i), n(i), r(i+1)] );
        A = s(1:r(i+1),1:r(i+1)) * v(:,1:r(i+1))';
    end

    %last core
    U{d} = reshape(A, [r(d), n(d), 1]);

    x = TTeMPS( U );
end
