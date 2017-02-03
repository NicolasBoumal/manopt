function M = complexcirclefactory(n)
% Returns a manifold struct to optimize over unit-modulus complex numbers.
%
% function M = complexcirclefactory()
% function M = complexcirclefactory(n)
%
% Description of vectors z in C^n (complex) such that each component z(i)
% has unit modulus. The manifold structure is the Riemannian submanifold
% structure from the embedding space R^2 x ... x R^2, i.e., the complex
% circle is identified with the unit circle in the real plane.
%
% By default, n = 1.
%
% See also spherecomplexfactory

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
%
%   July 7, 2014 (NB): Added ehess2rhess function.
%
%   Sep. 3, 2014 (NB): Correction to the dist function (extract real part).
%
%   April 13, 2015 (NB): Fixed logarithm.
%
%   Oct. 8, 2016 (NB)
%       Code for exponential was simplified to only treat the zero vector
%       as a particular case.
    
    if ~exist('n', 'var')
        n = 1;
    end

    M.name = @() sprintf('Complex circle (S^1)^%d', n);
    
    M.dim = @() n;
    
    M.inner = @(z, v, w) real(v'*w);
    
    M.norm = @(x, v) norm(v);
    
    M.dist = @(x, y) norm(acos(real(conj(x) .* y)));
    
    M.typicaldist = @() pi*sqrt(n);
    
    M.proj = @(z, u) u - real( conj(u) .* z ) .* z;	
    
    M.tangent = M.proj;
    
    % For Riemannian submanifolds, converting a Euclidean gradient into a
    % Riemannian gradient amounts to an orthogonal projection.
	M.egrad2rgrad = M.proj;
    
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(z, egrad, ehess, zdot)
        rhess = M.proj(z, ehess - real(z.*conj(egrad)).*zdot);
    end
    
    M.exp = @exponential;
    function y = exponential(z, v, t)
        
        if nargin == 2
            % t = 1;
            tv = v;
        else
            tv = t*v;
        end

        y = zeros(n, 1);

        nrm_tv = abs(tv);
        
        % We need to be careful for zero steps.
        mask = (nrm_tv > 0);
        y(mask) = z(mask).*cos(nrm_tv(mask)) + ...
                  tv(mask).*(sin(nrm_tv(mask))./nrm_tv(mask));
        y(~mask) = z(~mask);
        
    end
    
    M.retr = @retraction;
    function y = retraction(z, v, t)
        if nargin == 2
            % t = 1;
            tv = v;
        else
            tv = t*v;
        end
        y = sign(z+tv);
    end

    M.log = @logarithm;
    function v = logarithm(x1, x2)
        v = M.proj(x1, x2 - x1);
        di = real(acos(real(conj(x1) .* x2)));
        nv = abs(v);
        factors = di ./ nv;
        factors(di <= 1e-6) = 1;
		v = v .* factors;
    end
    
    M.hash = @(z) ['z' hashmd5( [real(z(:)) ; imag(z(:))] ) ];
    
    M.rand = @random;
    function z = random()
        z = sign(randn(n, 1) + 1i*randn(n, 1));
    end
    
    M.randvec = @randomvec;
    function v = randomvec(z)
        % i*z(k) is a basis vector of the tangent vector to the k-th circle
        v = randn(n, 1) .* (1i*z);
        v = v / norm(v);
    end
    
    M.lincomb = @matrixlincomb;
    
    M.zerovec = @(x) zeros(n, 1);
    
    M.transp = @(x1, x2, d) M.proj(x2, d);
    
    M.pairmean = @pairmean;
    function z = pairmean(z1, z2)
        z = sign(z1+z2);
    end

    M.vec = @(x, u_mat) [real(u_mat) ; imag(u_mat)];
    M.mat = @(x, u_vec) u_vec(1:n) + 1i*u_vec((n+1):end);
    M.vecmatareisometries = @() true;

end
