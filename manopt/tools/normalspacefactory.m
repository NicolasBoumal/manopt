function N = normalspacefactory(M, x)
% Returns a manifold structure representing the normal space to M at x.
%
% N = normalspacefactory(M, x)
%
% N defines a (linear) manifold that is the normal space to M at x. Points
% are represented as normal vectors to M at x. Normal vectors are also
% represented as normal vectors to M at x.
%
% This is chiefly useful to solve optimization problems involving normal
% vectors to M at x. The Riemannian (actually, Euclidean) structure on N
% is that of the normal space to M, so the inner product is inherited.
%
% See also: tangentspacefactory

% This file is part of Manopt: www.manopt.org.
% Original author: Jesus Briales, February 3, 2017.
% Contributors: 
% Change log: 

    % N is the manifold we build. y will be a point on N, thus also a
    % normal vector to M at x. This is a typical Euclidean space, hence it
    % will be easy to describe in terms of the tools available for M.
    N = struct();
    
    % u, u1 and u2 will be normal vectors to N at y. The normal space to
    % N at y is the normal space to M at x, thus u, u1 and u2 are also
    % normal vectors to M at x.
    
    N.dim   = @() numel(x) - M.dim(); % complement of tangent space
    N.inner = @(y, u1, u2) M.inner(x, u1, u2);
    N.norm  = @(y, u) M.norm(x, u);
    N.proj  = @projection;
  	function u_p = projection(y, u)
      % Projection into N, which is normal space of M, is complementary to projection onto tangent space
      u_p = u - M.proj(x,u);
    end
    N.typicaldist = @() N.dim();
    N.tangent = N.proj;
    N.egrad2rgrad = N.proj;
    N.ehess2rhess = @(y, eg, eh, d) N.proj(y,eh);
    N.exp = @exponential;
    N.retr = @exponential;
    N.log = @(y1, y2) M.lincomb(x, 1, y2, -1, y1);
    N.pairmean = @(y1, y2) M.lincomb(x, 0.5, y1, 0.5, y2);
    N.rand = @() randomvec( );
    N.randvec = @(y) randomvec( );
    function u = randomvec( )
      % create random vector with the appropriate dimensions
      u = randn(size(M.randvec(x)));
      % project that vector to the normal space
      u = projection([],u);
      % normalize with norm
      u = u / M.norm(x, u);
    end
    N.zerovec = M.zerovec;
    N.lincomb = M.lincomb;
    N.transp = @(y1, y2, u) u;
    N.hash = @(y) ['z' hashmd5(M.vec(x, y))];
    
    % In a Euclidean space, the exponential is merely the sum: y + tu.
    function yy = exponential(y, u, t)
        if nargin == 2
            t = 1;
        end
        yy = M.lincomb(x, 1, y, t, u);
    end
    
end
