function M = obliquefactory(n, m, transposed)
% Returns a manifold struct to optimize over matrices w/ unit-norm columns.
%
% function M = obliquefactory(n, m)
% function M = obliquefactory(n, m, transposed)
%
% Oblique manifold: deals with matrices of size n x m such that each column
% has unit 2-norm, i.e., is a point on the unit sphere in R^n. The metric
% is such that the oblique manifold is a Riemannian submanifold of the
% space of nxm matrices with the usual trace inner product, i.e., the usual
% metric.
%
% If transposed is set to true (it is false by default), then the matrices
% are transposed: a point Y on the manifold is a matrix of size m x n and
% each row has unit 2-norm. It is the same geometry, just a different
% representation.
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
%       Added tangent2ambient/tangent2ambient_is_identity pair.


    if ~exist('transposed', 'var') || isempty(transposed)
        transposed = false;
    end

    if transposed
        trnsp = @(X) X';
    else
        trnsp = @(X) X;
    end

    M.name = @() sprintf('Oblique manifold OB(%d, %d)', n, m);

    M.dim = @() (n-1)*m;

    M.inner = @(x, d1, d2) d1(:)'*d2(:);

    M.norm = @(x, d) norm(d(:));

    M.dist = @(x, y) norm(real(2*asin(.5*vecnorm(trnsp(x - y)))));

    M.typicaldist = @() pi*sqrt(m);

    M.proj = @(X, U) trnsp(projection(trnsp(X), trnsp(U)));

    M.tangent = M.proj;

    % For Riemannian submanifolds, converting a Euclidean gradient into a
    % Riemannian gradient amounts to an orthogonal projection.
    M.egrad2rgrad = M.proj;

    M.tangent2ambient_is_identity = true;
    M.tangent2ambient = @(X, U) U;

    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, U)
        X = trnsp(X);
        egrad = trnsp(egrad);
        ehess = trnsp(ehess);
        U = trnsp(U);

        PXehess = projection(X, ehess);
        inners = sum(X.*egrad, 1);
        rhess = PXehess - U .* inners;

        rhess = trnsp(rhess);
    end

    M.exp = @exponential;
    % Exponential on the oblique manifold
    function y = exponential(x, d, t)
        x = trnsp(x);
        d = trnsp(d);

        if nargin < 3
            % t = 1;
            td = d;
        else
            td = t*d;
        end

        nrm_td = vecnorm(td);

        y = x .* cos(nrm_td) + td .* sinxoverx(nrm_td);

        y = trnsp(y);
    end

    M.log = @logarithm;
    function v = logarithm(x1, x2)
        x1 = trnsp(x1);
        x2 = trnsp(x2);

        v = projection(x1, x2 - x1);
        dists = real(2*asin(.5*vecnorm(x1 - x2)));
        norms = vecnorm(v);
        factors = dists./norms;
        % For very close points, dists is almost equal to norms, but
        % because they are both almost zero, the division above can return
        % NaN's. To avoid that, we force those ratios to 1.
        factors(dists <= 1e-10) = 1;
        v = v .* factors;

        v = trnsp(v);
    end

    M.retr = @retraction;
    % Retraction on the oblique manifold
    function y = retraction(x, d, t)
        x = trnsp(x);
        d = trnsp(d);

        if nargin < 3
            % t = 1;
            td = d;
        else
            td = t*d;
        end

        y = normalize_columns(x + td);

        y = trnsp(y);
    end

    % Inverse retraction: see spherefactory.m for background
    M.invretr = @inverse_retraction;
    function d = inverse_retraction(x, y)

        x = trnsp(x);
        y = trnsp(y);

        d = (y .* (1./sum(x.*y, 1))) - x;

        d = trnsp(d);

    end


    M.hash = @(x) ['z' hashmd5(x(:))];

    M.rand = @() trnsp(normalize_columns(randn(n, m)));

    M.randvec = @randvec;
    function d = randvec(x)
        d = trnsp(projection(x, randn(n, m)));
        d = d / norm(d(:));
    end

    M.lincomb = @matrixlincomb;

    M.zerovec = @(x) trnsp(zeros(n, m));

    M.transp = @(x1, x2, d) M.proj(x2, d);

    M.pairmean = @pairmean;
    function y = pairmean(x1, x2)
        y = trnsp(x1+x2);
        y = normalize_columns(y);
        y = trnsp(y);
    end

    % vec returns a vector representation of an input tangent vector which
    % is represented as a matrix. mat returns the original matrix
    % representation of the input vector representation of a tangent
    % vector. vec and mat are thus inverse of each other. They are
    % furthermore isometries between a subspace of R^nm and the tangent
    % space at x.
    vect = @(X) X(:);
    M.vec = @(x, u_mat) vect(trnsp(u_mat));
    M.mat = @(x, u_vec) trnsp(reshape(u_vec, [n, m]));
    M.vecmatareisometries = @() true;

end

% Given a matrix X, returns the same matrix but with each column scaled so
% that they have unit 2-norm.
% This is equivalent to a built-in call to:  normalize(X, 'norm', 2) 
% but on a quick profiler test in 2023 the code below was faster.
function X = normalize_columns(X)
    % The following is faster than "norms(X, 2, 1)".
    % It's sometimes faster and sometimes slower than "sqrt(sum(X.^2, 1))".
    nrms = vecnorm(X);
    % The following is faster than "X ./ nrms", though a tad less accurate.
    % It's also much faster than "bsxfun(@times, X, 1./nrms)".
    X = X .* (1 ./ nrms);
end

% Orthogonal projection of the ambient vector H to the tangent space at X.
% This operates on colums.
function PXH = projection(X, H)

    % Compute the inner product between each vector H(:, i) with its root
    % point X(:, i), that is, X(:, i)' * H(:, i). Returns a row vector.
    inners = sum(X.*H, 1);

    % Subtract from H the components of the H(:, i)'s that are parallel to
    % the root points X(:, i).
    PXH = H - X .* inners;

    % % Equivalent but slow code:
    % m = size(X, 2);
    % PXH = zeros(size(H));
    % for i = 1 : m
    %     PXH(:, i) = H(:, i) - X(:, i) * (X(:, i)'*H(:, i));
    % end

end

