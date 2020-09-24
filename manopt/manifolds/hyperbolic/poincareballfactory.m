function M = poincareballfactory(k, n, gpuflag)
% Factory for matrices whose columns live on the Poincare ball manifold
%
% function M = poincareballfactory(k, n)
%
% Set gpuflag = true to have points, tangent vectors and ambient vectors
% stored on the GPU. If so, computations can be done on the GPU directly.


    if ~exist('n', 'var') || isempty(n)
        n = 1;
    end
    
    if ~exist('gpuflag', 'var') || isempty(gpuflag)
        gpuflag = false;
    end
    
    % If gpuflag is active, new arrays (e.g., via rand, randn, zeros, ones)
    % are created directly on the GPU; otherwise, they are created in the
    % usual way (in double precision).
    if gpuflag
        array_type = 'gpuArray';
    else
        array_type = 'double';
    end
        

    if n == 1
        M.name = @() sprintf('Poincare ball B_%d', k);
    else
        M.name = @() sprintf('Poincare ball B_%d^%d', k, n);
    end
    
    M.dim = @() k * n;
    
    M.conformal_factor = @(x) 2 ./ (1 - sum(x .* x, 1));
    
    M.inner = @(x, d1, d2) sum((d1 .* d2) * (M.conformal_factor(x).^2));
    
    M.norm = @(x, d) sqrt(M.inner(x, d, d));
    
    M.dist = @dist;
    function d = dist(x, y)
        norms2x = sum(x .* x, 1);
        norms2y = sum(y .* y, 1);
        norms2diff = sum((x - y) .* (x - y), 1);
        d = sqrt(sum(acosh(1 + 2 * norms2diff ./ (1 - norms2x) ./ (1 - norms2y)) .^ 2));
    end

    M.typicaldist = @() M.dim / 8;
    
    % Identity map since the embedding space is the tangent space.
    M.proj = @(x, d) d;
    
    M.tangent = M.proj;
    
    % The Poincaré ball is not a Riemannian submanifold and the
    % Euclidean gradient cannot simply be projected.
    M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(x, egrad)
        factor = M.conformal_factor(x);
        rgrad = egrad .* ((1 ./ factor).^2);
    end
    
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(x, egrad, ehess, u)
        factor = M.conformal_factor(x);
        rhess = (u .* sum(egrad .* x, 1) - egrad .* sum(u .* x, 1) - x .* sum(u .* egrad, 1) + ehess ./ factor) ./factor;
    end

    M.mobius_addition = @mobius_addition;
    function res = mobius_addition(x, y)
        sp = sum(x .* y, 1);
        norm2x = sum(x .* x, 1);
        norm2y = sum(y .* y, 1);
        res = (x .* (1 + 2 .* sp + norm2y) + y .* (1 - norm2x)) ./ (1 + 2 .* sp + norm2x .* norm2y);
    end

    M.exp = @exponential;
    
    M.retr = @retraction;
    M.invretr = @inverse_retraction;
    
    M.log = @logarithm;
    
    M.hash = @(x) ['z' hashmd5(x(:))];
    
    % Columns are sampled uniformly at random in the unit ball.
    M.rand = @() sample_ball_uniformly(k, n);
    
    M.randvec = @(x) randn(k, n);
    
    M.zerovec = @(x) zeros(k, n);
    
    M.pairmean = @pairmean;
    function y = pairmean(x1, x2)
        y = M.exp(x1, M.log(x1, x2) / 2);
    end

    M.vec = @(x, u_mat) u_mat(:);
    M.mat = @(x, u_vec) reshape(u_vec, [k, n]);
    M.vecmatareisometries = @() true;
    
    
    % Automatically convert a number of tools to support GPU.
    if gpuflag
        M = factorygpuhelper(M);
    end

end

function z = mobius_addition(x, y)
    inner = sum(x .* y, 1);
    norms2x = sum(x .* x, 1);
    norms2y = sum(y .* y, 1);
    z = ((1 + 2 * inner + norms2y) .* x + (1 - norms2x) .* y ) ./ (1 + 2 * inner + norms2x .* norms2y);
end

% Exponential on the Poincaré ball.
function y = exponential(x, d, t)
    if nargin == 2
        % t = 1
        td = d;
    else
        td = t*d;
    end
    
    normstd = vecnorm(td);
    factor = (1 - sum(x .* x, 1));
    % Avoid dividing by zero.
    w = td .* (tanh(normstd ./ factor) ./ (normstd + (normstd == 0)));
    y = mobius_addition(x, w);
end

% Exponential map is cheap so use it as a retraction.
function y = retraction(x, d, t)
    if nargin == 2
        % t = 1;
        td = d;
    else
        td = t*d;
    end
    y = exponential(x, td);
end

% Given x and y two points on the manifold, if there exists a tangent
% vector d at x such that Retr_x(d) = y, this function returns d.
function d = inverse_retraction(x, y)
    d = logarithm(x, y);
end

function v = logarithm(x, y)
    w = mobius_addition(-x, y);
    normsw = vecnorm(w);
    factor = 1 - sum(x .* x, 1);
    v = w .* factor .* atanh(normsw) ./ normsw;
end

function x = sample_ball_uniformly(k, n)
    isotropic = randn(k, n);
    isotropic = isotropic ./ vecnorm(isotropic);
    radiuses = rand(1, n) .^ (1 / k);
    x = isotropic .* radiuses;
end