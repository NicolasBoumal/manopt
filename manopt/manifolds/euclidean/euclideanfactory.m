function M = euclideanfactory(m, n)
% Returns a manifold struct to optimize over m-by-n matrices.
%
% function M = euclideanfactory(m, n)
%
% Returns M, a structure describing the Euclidean space of m-by-n matrices,
% equipped with the standard Frobenius distance and associated trace inner
% product, as a manifold for Manopt.
%
% m and n in general can be vectors to handle multidimensional arrays.
% If either of m or n is a vector, they are concatenated as [m n].
%
% Using this 'manifold', Manopt can be used to solve standard
% unconstrained optimization problems, for example in replacement of
% Matlab's fminunc.
%
% See also: euclideancomplexfactory

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: Bamdev Mishra, May 4, 2015.
% Change log: 
%
%   July 5, 2013 (NB):
%       Added egred2rgrad, ehess2rhess, mat, vec, tangent.
%   May 4, 2015 (BM):
%       Added functionality to handle multidimensional arrays.


    if ~exist('n', 'var') || isempty(n)
        n = 1;
    end
    
    if ~isinteger(m) || ~isinteger(n)
        error('Bad usage of euclideanfactory');
    end
    
    if size(m, 1) >= size(m, 2)
        m = m'; % Make #columns bigger than #rows.
    end
    
    if size(n, 1) >= size(n, 2)
        n = n'; % Make #columns bigger than #rows.
    end
    
    if size(m, 1) > 1 || size(n, 1) > 1
        % m or n being a matrix is not what we should be handling here.
        error('Bad usage of euclideanfactory');  
    end
    
    % We have no reason to not believe in the scenario below.
    dimensions_vec = [m, n]; % We have a row vector.
    
    
    M.name = @() sprintf('Euclidean space R^(%s)', num2str(dimensions_vec)); % BM: okay.
    
    M.dim = @() prod(dimensions_vec);% BM: replacing m*n;
    
    M.inner = @(x, d1, d2) d1(:).'*d2(:); % BM: okay.
    
    M.norm = @(x, d) norm(d(:), 'fro');% BM: replacing norm(d, 'fro');
    
    M.dist = @(x, y) norm(x(:) - y(:), 'fro');% BM: replacing norm(x-y, 'fro');
    
    M.typicaldist = @() sqrt(prod(dimensions_vec));% BM: replacing sqrt(m*n);
    
    M.proj = @(x, d) d; % BM: okay.
    
    M.egrad2rgrad = @(x, g) g; % BM: okay.
    
    M.ehess2rhess = @(x, eg, eh, d) eh; % BM: okay.
    
    M.tangent = M.proj; 
    
    M.exp = @exp;
    function y = exp(x, d, t)
        if nargin == 3
            y = x + t*d; % BM: okay.
        else
            y = x + d; % BM: okay.
        end
    end
    
    M.retr = M.exp;
	
	M.log = @(x, y) y-x; % BM: okay.

    M.hash = @(x) ['z' hashmd5(x(:))]; % BM: okay.
    
    M.rand = @() randn(dimensions_vec);% BM: replacing randn(m, n);
    
    M.randvec = @randvec;
    function u = randvec(x) %#ok<INUSD>
        u = randn(dimensions_vec);% BM: replacing randn(m, n);
        u = u / norm(u(:), 'fro');% BM: replacing u / norm(u, 'fro');
    end
    
    M.lincomb = @lincomb;
    function v = lincomb(x, a1, d1, a2, d2) %#ok<INUSL>
        if nargin == 3
            v = a1*d1;
        elseif nargin == 5
            v = a1*d1 + a2*d2;
        else
            error('Bad usage of euclidean.lincomb');
        end
    end
    
    M.zerovec = @(x) zeros(dimensions_vec);% BM: replacing zeros(m, n);
    
    M.transp = @(x1, x2, d) d;
    
    M.pairmean = @(x1, x2) .5*(x1+x2); % BM: okay.
    
    M.vec = @(x, u_mat) u_mat(:); % BM: okay.
    M.mat = @(x, u_vec) reshape(u_vec, dimensions_vec);% BM: replacing reshape(u_vec, [m, n]);
    M.vecmatareisometries = @() true;

end
