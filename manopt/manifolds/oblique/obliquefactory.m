function M = obliquefactory(n, m, dirflag, gpuflag)
% Returns a manifold struct to optimize on matrices w/ unit-norm cols/rows.
%
% function M = obliquefactory(n, m)
% function M = obliquefactory(n, m, 'cols')
% function M = obliquefactory(n, m, 'rows')
% function M = obliquefactory(n, m, dirflag, 'gpu')
%
% The oblique manifold is a product of spheres, embedded in R^(nxm).
%
% By default, columns have unit norm (product of m spheres in R^n).
% This can also be expressed with the flag 'cols'.
% 
% If that flag is set to 'rows', then the rows have unit norm instead
% (product of n spheres in R^m).
% 
% The metric is such that the oblique manifold is a Riemannian submanifold
% of the space of nxm matrices with the trace inner product (Frobenius).
%
% Set gpuflag  to 'gpu' to have points, tangent vectors and ambient vectors
% stored on the GPU. If so, computations in cost, grad etc. can be done on
% the GPU directly. By default, the GPU is not used. This can also be
% expressed with the flag 'nogpu'.
%
%
% For backward compatibility, the two input flags can also be boolean:
%   obliquefactory(n, m, false) == obliquefactory(n, m, 'cols')
%   obliquefactory(n, m, true) == obliquefactory(m, n, 'rows') (mind m<->n)
%   gpuflag: true == 'gpu' and false == 'nogpu'.
%   
%
% See also: spherefactory obliquecomplexfactory

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors:
% Change log:
%
%   July 16, 2013 (NB)
%       Added 'transposed' option, mainly for ease of comparison with the
%       elliptope geometry.
%
%   Nov. 29, 2013 (NB)
%       Added normalize_columns function to make it easier to exploit the
%       bsxfun formulation of column normalization, which avoids using for
%       loops and provides performance gains. The exponential still uses a
%       for loop.
%
%   April 4, 2015 (NB)
%       Log function modified to avoid NaN's appearing for close by points.
%
%   April 13, 2015 (NB)
%       Exponential now without for-loops.
%
%   Oct. 8, 2016 (NB)
%       Code for exponential was simplified to only treat the zero vector
%       as a particular case.
%
%   Oct. 21, 2016 (NB)
%       Bug caught in M.log: the function called v = M.proj(x1, x2 - x1),
%       which internally applies transp to inputs and outputs. But since
%       M.log had already taken care of transposing things, this introduced
%       a bug (which only triggered if using M.log in transposed mode.)
%       The code now calls "v = projection(x1, x2 - x1);" since projection
%       assumes the inputs and outputs do not need to be transposed.
%
%   July 20, 2017 (NB)
%       Distance function is now accurate for close-by points. See notes
%       inside the spherefactory file for details. Also improves distance
%       computations as part of the log function.
%
%   Sep. 24, 2023 (NB)
%       Edited bsxfun out, in favor of more modern .* and ./ syntax
%       operating on a matrix and a row/col vector. This is much faster.
%       Also replaced manual computations of the type sqrt(sum(X.^2, 1)) by
%       the built-in call vecnorm(X). That isn't necessarily always faster,
%       but it seems to be reliably close in terms of performance, and a
%       built-in function has a chance to become better in future releases.
%
%   Sep. 24, 2023 (NB)
%       Added GPU support: just set gpuflag = true.
%
%   Sep. 24, 2023 (NB)
%       Added tangent2ambient/tangent2ambient_is_identity pair.
%
%   June 26, 2024 (NB)
%       Modified the way input flags work, for legibility.
%       Improved performance for transposed case.
%       Exposed M.normalize(Y), which normalizes the colums or the rows so
%       as to offer metric projection to the manifold.


    % Implementation notes:
    % - vecnorm(X, 2, dim) seems faster than norms(X, 2, dim).
    %   It is sometimes faster and sometimes slower
    %   than sqrt(sum(X.^2, dim)).
    % - normalize() (below) seems faster than the built-in
    %   normalize(X, 'norm', 2), based on a quick profiler test in 2023.


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

    if unitcols
        name = sprintf('Oblique manifold OB(%d, %d), unit columns', n, m);
    else
        name = sprintf('Oblique manifold OB(%d, %d), unit rows', n, m);
    end
    M.name = @() name;

    if unitcols
        Mdim = (n-1)*m;
    else
        Mdim = (m-1)*n;
    end
    M.dim = @() Mdim;

    M.inner = @(X, U, V) U(:).'*V(:);

    M.norm = @(X, U) norm(U(:));

    M.dist = @(X, Y) norm(real(2*asin(.5*vecnorm(X - Y, 2, dim))));

    if unitcols
        typicaldist = pi*sqrt(m);
    else
        typicaldist = pi*sqrt(n);
    end
    M.typicaldist = @() typicaldist;

    M.proj = @projection;
    % Orthogonal projection of H in R^(nxm) to the tangent space at X.
    function PXH = projection(X, H)
        % Compute the inner product between each column/row of H
        % with the corresponding column/row of X.
        inners = sum(X.*H, dim);
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
        inners = sum(X.*egrad, dim);
        rhess = projection(X, ehess - U.*inners);
    end

    M.exp = @exponential;
    % Exponential on the oblique manifold
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
        % For very close points, dists is almost equal to norms, but
        % because they are both almost zero, the division above can return
        % NaN's. To avoid that, we force those ratios to 1.
        factors(dists <= 1e-10) = 1;
        V = V .* factors;
    end

    M.normalize = @normalize;
    % Scale each col/row of X by its norm.
    function Y = normalize(X)
        nrms = vecnorm(X, 2, dim);
        % The following is faster than "X ./ nrms", though a tad less
        % accurate. It's also much faster than bsxfun(@times, X, 1./nrms).
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

    % Inverse retraction: see spherefactory.m for background
    M.invretr = @inverse_retraction;
    function U = inverse_retraction(X, Y)
        U = (Y .* (1./sum(X.*Y, dim))) - X;
    end

    % The retraction is second order.
    M.retr2 = M.retr;

    M.hash = @(x) ['z' hashmd5(x(:))];

    M.rand = @() normalize(randn(n, m, array_type));

    M.randvec = @randvec;
    function U = randvec(X)
        U = projection(X, randn(n, m, array_type));
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
    M.vec = @(X, U_mat) U_mat(:);
    M.mat = @(X, U_vec) reshape(U_vec, [n, m]);
    M.vecmatareisometries = @() true;

    % Automatically convert a number of tools to support GPU.
    if usegpu
        M = factorygpuhelper(M);
    end

end
