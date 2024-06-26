function M = obliquecomplexfactory(n, m, dirflag, gpuflag)
% A manifold struct to optimize on complex matrices w/ unit-norm cols/rows.
%
% function M = obliquecomplexfactory(n, m)
% function M = obliquecomplexfactory(n, m, 'cols')
% function M = obliquecomplexfactory(n, m, 'rows')
% function M = obliquecomplexfactory(n, m, dirflag, 'gpu')
%
% The oblique manifold is a product of spheres, embedded in C^(nxm).
%
% By default, columns have unit norm (product of m spheres in C^n).
% This can also be expressed with the flag 'cols'.
% 
% If that flag is set to 'rows', then the rows have unit norm instead
% (product of n spheres in C^m).
% 
% The metric is such that the oblique manifold is a Riemannian submanifold
% of the space of nxm matrices with the trace inner product (Frobenius):
%
%   <A, B> = real(trace(A'*B))
%
% Effectively, this treats C^n as R^(2n), with the real and imaginary parts
% treated separately as 2n real coordinates.
%
% Set gpuflag  to 'gpu' to have points, tangent vectors and ambient vectors
% stored on the GPU. If so, computations in cost, grad etc. can be done on
% the GPU directly. By default, the GPU is not used. This can also be
% expressed with the flag 'nogpu'.
%
%
% For backward compatibility, the two input flags can also be boolean:
%   obliquecomplexfactory(n, m, false)
%       == obliquecomplexfactory(n, m, 'cols')
%   obliquecomplexfactory(n, m, true)
%       == obliquecomplexfactory(m, n, 'rows') (mind m <-> n)
%   gpuflag: true == 'gpu' and false == 'nogpu'.
%   
%
% Note: obliquecomplexfactory(n, 1, 'rows') is equivalent to (but
% potentially slower than) complexcirclefactory(n).
%
% See also: spherecomplexfactory complexcirclefactory obliquefactory

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Sep. 3, 2014.
% Contributors:
% Change log:
%
%   Oct. 21, 2016 (NB)
%       Formatted for inclusion in Manopt release.
%
%   July 20, 2017 (NB)
%       Distance function is now accurate for close-by points. See notes
%       inside the spherefactory file for details. Also improves distances
%       computation as part of the log function.
%
%   May 28, 2023 (NB)
%       Fixed bug in M.log in case 'transposed' is true (bug reported by
%       Lingping Kong).
%
%   June 26, 2024 (NB)
%       Revamped to match the new format of the real version of this
%       factory. In the process, we get GPU support. Backward compatible.


    if ~exist('dirflag', 'var') || isempty(dirflag)
        dirflag = 'cols';
    end
    if islogical(dirflag)   % legacy support for Manopt <= 7.1.
        if dirflag          % dirflag was a Boolean called 'transposed'.
            [n, m] = deal(m, n);
            dirflag = 'rows';
        else
            dirflag = 'cols';
        end
    end
    switch lower(dirflag)
        case 'cols'
            unitcols = true;
            dim = 1; % for use with vecnorm and sum, to operate on columns
        case 'rows'
            unitcols = false;
            dim = 2; % for use with vecnorm and sum, to operate on rows
        otherwise
            error('The direction flag should be ''cols'' or ''rows''.');
    end


    if ~exist('gpuflag', 'var') || isempty(gpuflag)
        gpuflag = 'nogpu';
    end
    if islogical(gpuflag)   % legacy support for Manopt <= 7.1.
        if gpuflag          % gpuflag was a Boolean.
            gpuflag = 'gpu';
        else
            gpuflag = 'nogpu';
        end
    end
    switch lower(gpuflag)
        case 'gpu'
            usegpu = true;
        case 'nogpu'
            usegpu = false;
        otherwise
            error('The GPU flag should be ''gpu'' or ''nogpu''.');
    end



    % If gpuflag is active, new arrays (e.g., via rand, randn, zeros, ones)
    % are created directly on the GPU; otherwise, they are created in the
    % usual way (in double precision).
    if usegpu
        array_type = 'gpuArray';
    else
        array_type = 'double';
    end

    basename = 'Complex oblique manifold COB(%d, %d), ';
    if unitcols
        name = sprintf([basename, 'unit columns'], n, m);
    else
        name = sprintf([basename, 'unit rows'], n, m);
    end
    M.name = @() name;

    if unitcols
        Mdim = (2*n-1)*m;
    else
        Mdim = (2*m-1)*n;
    end
    M.dim = @() Mdim;

    M.inner = @(X, U, V) real(U(:)'*V(:));

    M.norm = @(X, U) norm(U(:));

    M.dist = @(X, Y) norm(real(2*asin(.5*vecnorm(X - Y, 2, dim))));

    if unitcols
        typicaldist = pi*sqrt(m);
    else
        typicaldist = pi*sqrt(n);
    end
    M.typicaldist = @() typicaldist;

    M.proj = @projection;
    % Orthogonal projection of H in C^(nxm) to the tangent space at X.
    function PXH = projection(X, H)
        % Compute the inner product between each column/row of H
        % with the corresponding column/row of X.
        inners = real(sum(conj(X).*H, dim));
        % Remove from H the components that are parallel to X, by row/col.
        PXH = H - X.*inners;
    end

    M.tangent = M.proj;

    % For Riemannian submanifolds, converting a Euclidean gradient into a
    % Riemannian gradient amounts to an orthogonal projection.
    M.egrad2rgrad = M.proj;

    M.tangent2ambient_is_identity = true;
    M.tangent2ambient = @(X, U) U;

    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, U)
        inners = real(sum(conj(X).*egrad, dim));
        rhess = projection(X, ehess - U.*inners);
    end

    M.exp = @exponential;
    function Y = exponential(X, U, t)
        if nargin < 3  % t = 1;
            tU = U;
        else
            tU = t*U;
        end
        nrm_tU = vecnorm(tU, 2, dim);
        Y = X .* cos(nrm_tU) + tU .* sinxoverx(nrm_tU);
    end

    M.log = @logarithm;
    function V = logarithm(X1, X2)
        difference = X2 - X1;
        dists = real(2*asin(.5*vecnorm(difference, 2, dim)));
        V = projection(X1, difference);
        norms = vecnorm(V, 2, dim);
        factors = dists./norms;
        factors(dists <= 1e-10) = 1;
        V = V .* factors;
    end

    M.normalize = @normalize;
    % Scale each col/row of X by its norm.
    function Y = normalize(X)
        nrms = vecnorm(X, 2, dim);
        Y = X .* (1 ./ nrms);
    end

    M.retr = @retraction;
    % Metric projection retraction
    function Y = retraction(X, U, t)
        if nargin < 3
            tU = U;  % t = 1;
        else
            tU = t*U;
        end
        Y = normalize(X + tU);
    end

    % Inverse retraction
    M.invretr = @inverse_retraction;
    function U = inverse_retraction(X, Y)
        U = (Y .* (1./real(sum(conj(X).*Y, dim)))) - X;
    end

    M.hash = @(x) ['z' hashmd5([real(x(:)) ; imag(x(:))])];

    M.rand = @() normalize(   randn(n, m, array_type) + ...
                           1i*randn(n, m, array_type));

    M.randvec = @randvec;
    function U = randvec(X)
        randmat = randn(n, m, array_type) + 1i*randn(n, m, array_type);
        U = projection(X, randmat);
        U = U / norm(U(:));
    end

    M.lincomb = @matrixlincomb;

    M.zerovec = @(x) zeros(n, m, array_type);

    M.transp = @(X1, X2, U) projection(X2, U);

    M.pairmean = @(X1, X2) normalize(X1+X2);

    % vec returns a vector representation of an input tangent vector which
    % is represented as a matrix.
    % mat returns the original matrix representation of the input vector
    % representation of a tangent vector.
    % vec and mat are thus inverse of each other. They are furthermore
    % isometries between a subspace of R^nm and the tangent space at X.
    M.vec = @(X, U_mat) [real(U_mat(:)) ; imag(U_mat(:))];
    M.mat = @(X, U_vec)    reshape(U_vec(1:n*m), [n, m]) + ...
                        1i*reshape(U_vec((n*m+1):end), [n, m]);
    M.vecmatareisometries = @() true;

    % Automatically convert a number of tools to support GPU.
    if usegpu
        M = factorygpuhelper(M);
    end

end
