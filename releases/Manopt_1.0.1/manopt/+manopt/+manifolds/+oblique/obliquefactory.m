function M = obliquefactory(n, m)
% Returns a manifold struct to optimize over matrices w/ unit-norm columns.
%
% function M = obliquefactory(n, m)
%
% Oblique manifold: deals with matrices of size n x m such that each column
% has unit 2-norm, i.e., is a point on the unit sphere in R^n. The metric
% is such that the oblique manifold is a Riemannian submanifold of the
% space of nxm matrices with the usual trace inner product, i.e., the usual
% metric.
%
% See also: spherefactory

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 


    % Import necessary tools etc. here
    import manopt.privatetools.hashmd5;

    M.name = @() sprintf('Oblique manifold OB(%d, %d)', n, m);
    
    M.dim = @() (n-1)*m;
    
    M.inner = @(x, d1, d2) d1(:).'*d2(:);
    
    M.norm = @(x, d) norm(d(:));
    
    M.dist = @(x, y) norm(real(acos(sum(x.*y, 1))));
    
    M.typicaldist = @() pi*sqrt(m);
    
    M.proj = @projection;
    
    % For Riemannian submanifolds, converting a Euclidean gradient into a
    % Riemannian gradient amounts to an orthogonal projection.
	M.egrad2rgrad = M.proj;
    
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, U)
        PXehess = projection(X, ehess);
        inners = sum(X.*egrad, 1);
        rhess = PXehess - bsxfun(@times, U, inners);
    end
    
    M.exp = @exponential;

    M.log = @logarithm;
    function v = logarithm(x1, x2)
        % TODO: this is probably not a good way to proceed, is it?
        v = M.proj(x1, x2 - x1);
        dists = acos(sum(x1.*x2, 1));
        norms = sqrt(sum(v.^2, 1));
        for i = 1 : m
            if dists(i) > 1e-6
                v(:, i) = v(:, i) * (dists(i)/norms(i));
            end
        end
    end

    M.retr = @retraction;

    M.hash = @(x) ['z' hashmd5(x(:))];
    
    M.rand = @() random(n, m);
    
    M.randvec = @(x) randomvec(n, m, x);
    
    M.lincomb = @lincomb;
    
    M.zerovec = @(x) zeros(n, m);
    
    M.transp = @(x1, x2, d) M.proj(x2, d);
    
    M.pairmean = @pairmean;
    function y = pairmean(x1, x2)
        y = x1+x2;
        y = y ./ repmat(sqrt(sum(y.^2, 1)), n, 1);
    end

end

function PXH = projection(X, H)
% Orthogonal projection of the ambient vector H onto the tangent space at X

    % Compute the inner product between each vector H(:, i) with its root
    % point X(:, i), that is, X(:, i).' * H(:, i). Returns a row vector.
    inners = sum(X.*H, 1);
    
    % Subtract from H the components of the H(:, i)'s that are parallel to
    % the roots points X(:, i).
    PXH = H - bsxfun(@times, X, inners);

    % % Equivalent but slow code:
    % m = size(X, 2);
    % PXH = zeros(size(H));
    % for i = 1 : m
    %     PXH(:, i) = H(:, i) - X(:, i) * (X(:, i)'*H(:, i));
    % end

end

% Exponential on the oblique manifold
function y = exponential(x, d, t)

    if nargin < 3
        t = 1.0;
    end

    m = size(x, 2);
    y = zeros(size(x));
    if t ~= 0
        for i = 1 : m
            y(:, i) = sphere_exponential(x(:, i), d(:, i), t);
        end
    else
        y = x;
    end

end

% Exponential on the sphere TODO: this is copied an pasted: bad practice!
function y = sphere_exponential(x, d, t)

    if nargin == 2
        t = 1.0;
    end
    
    td = t*d;
    
    nrm_td = norm(td);
    
    if nrm_td > 1e-6
        y = x*cos(nrm_td) + (td/nrm_td)*sin(nrm_td);
    else
        % if the step is too small, to avoid dividing by nrm_td, we choose
        % to approximate with this retraction-like step.
        y = x + td;
        y = y / norm(y);
    end

end

% Retraction on the oblique manifold
function y = retraction(x, d, t)

    if nargin < 3
        t = 1.0;
    end

    m = size(x, 2);
    y = zeros(size(x));
    if t ~= 0
        for i = 1 : m
            y(:, i) = x(:, i) + t*d(:, i);
            y(:, i) = y(:, i) / norm(y(:, i));
        end
    else
        y = x;
    end

end

% Uniform random sampling on the sphere.
function x = random(n, m)

    x = randn(n, m);
    x = x ./ repmat(sqrt(sum(x.^2)), n, 1);

end

% Random normalized tangent vector at x.
function d = randomvec(n, m, x)

    d = randn(n, m);
    d = projection(x, d);
    d = d / norm(d(:));

end

% Linear combination of tangent vectors
function d = lincomb(x, a1, d1, a2, d2) %#ok<INUSL>

    if nargin == 3
        d = a1*d1;
    elseif nargin == 5
        d = a1*d1 + a2*d2;
    else
        error('Bad use of oblique.lincomb.');
    end

end
