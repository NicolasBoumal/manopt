function M = hyperbolicfactory(n, m, transposed)
% Returns a manifold struct to optimize over matrices w/ hyperbolic column vectors.
%
% function M = hyperbolicfactory(n)
% function M = hyperbolicfactory(n, m)
% function M = hyperbolicfactory(n, m, transposed)
%
% Hyperbolic manifold: deals with matrices of size (n+1) x m such that each column
% is a point in the hyperbolic space with the Minkowski norm = -1, i.e., it is a point in R^(n+1). 
% The Minkowski inner product between two vectors x = [x1, x2, ..., x_{n+1}] and 
% y = [y1, y2, ..., y_{n+1}] is defined as -x1y1 + sum_{i = 2 to n+1} xiyi.
% The Minkowski metric is such that the hyperbolic manifold is a Riemannian submanifold of the
% pseudo-Riemannian ambient space R^(n+1). The geometry is called the hyperboloid or the Lorentz 
% geometry.
%
% If transposed is set to true (it is false by default), then the matrices
% are transposed: a point Y on the manifold is a matrix of size m x (n+1) and
% each row is an element in the hyperbolic space. It is the same geometry, just a different
% representation.
%
% Resources:
% 1. Nickel, Maximillian and Kiela, Douwe "Learning Continuous Hierarchies in the Lorentz Model of 
% Hyperbolic Geometry", ICML, 2018.
%
% 2. Wilson, Benjamin and Leimeister, Matthias, "Gradient descent in hyperbolic space", arXiv preprint
% arXiv:1805.08207 (2018).
% 
% 3. Pennec, Xavier, "Hessian of the Riemannian squared distance", HAL INRIA, 2017.
%
% Ported primarily from the McTorch toolbox at https://github.com/mctorch/mctorch.
%
% See also: spherefactory obliquefactory obliquecomplexfactory


% This file is part of Manopt: www.manopt.org.
% Original authors: Bamdev Mishra <bamdevm@gmail.com>, Mayank Meghwanshi, 
% Pratik Jawanpuria, Anoop Kunchukuttan, and Hiroyuki Kasai Oct 28, 2018.
% Contributors: 
% Change log:

    if ~exist('m', 'var') || isempty(m)
        m = 1;
    end 
    
    if ~exist('transposed', 'var') || isempty(transposed) % BM: okay
        transposed = false;
    end
    
    if transposed % BM: okay.
        trnsp = @(X) X';
    else
        trnsp = @(X) X;
    end

    M.name = @() sprintf('Hyperbolic manifold H(%d, %d)', n, m); % BM: okay
    
    M.dim = @() (n)*m; % BM: okay

    % For the Minkowski bilinear inner product.
    g = ones(n+1,1); % A column vector.
    g(1,1) = -1;
       
    M.inner = @myinner; % BM: okay
    function ip = myinner(x, d1, d2)
    	d1 = trnsp(d1);
    	d2 = trnsp(d2);
    	ip = minkowskiinner(d1, d2); % BM: arguments are column wise.
    end
    function ip = minkowskiinner(d1, d2) % BM: we assume that the arguments are column wise organized.
        d2 = bsxfun(@times, d2, g);
    	ip = d1(:)'*d2(:);
    end

    myeps = 0;  eps; % BM: okay.

    M.norm = @(x, d) sqrt(max(myinner(x, d, d), myeps)); % BM: okay to avoid negative entries.
    
    M.dist = @(x, y) norm((acosh(-sum(trnsp(x).*(bsxfun(@times, trnsp(y), g)), 1))));  % BM: okay.
        
    M.proj = @projection;  % BM: okay.
    function PU = projection(X, U)
    	X = trnsp(X);
    	U = trnsp(U);

    	XUinner = sum(X.*(bsxfun(@times, U, g)), 1); % A row vector.

        PU = U + bsxfun(@times, X, XUinner);

    	PU = trnsp(PU);

        % % Debug: to test whether the obtained vector is on the tangent space.
        % X = trnsp(X);
        % myinner(X, X, PU)  % BM: to prove that the vector belongs to the tangent space.
    end
    
    M.tangent = M.proj;  % BM: okay
    
    % For Riemannian submanifolds, converting the Euclidean gradient into the
    % Riemannian gradient amounts to an orthogonal projection. However, we first need to correct for the Minkowski bilinear inner product. 
    M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(X, egrad)
        X = trnsp(X);
        egrad = trnsp(egrad);

        egrad = bsxfun(@times, egrad, g); % BM: correcting for the Minkowski bilinear inner product.
        
        rgrad = projection(trnsp(X), trnsp(egrad)); 
    end
    
    M.ehess2rhess = @ehess2rhess; % BM: okay.
    function rhess = ehess2rhess(X, egrad, ehess, U)
        X = trnsp(X);
        egrad = trnsp(egrad);
        ehess = trnsp(ehess);
        U = trnsp(U);

        Xegradinner = sum(X.*egrad, 1); % A row vector.
        Xegradinnerdot = sum(X.*ehess, 1) + sum(U.*egrad, 1); % A row vector.
        rhess = bsxfun(@times, ehess, g) +bsxfun(@times, U, Xegradinner) + bsxfun(@times, X, Xegradinnerdot);

        rhess = projection(trnsp(X), trnsp(rhess));
    end
    
    M.exp = @exponential;
    % Exponential on the hyperbolic manifold
    function y = exponential(x, d, t) % BM: okay.
        x = trnsp(x);
        d = trnsp(d);
        
        if nargin < 3
            % t = 1;
            td = d;
        else
            td = t*d;
        end        

        nrm_td = sqrt(sum(td.*(bsxfun(@times, td, g)), 1)); % BM: okay.

        y = bsxfun(@times, x, cosh(nrm_td)) + ...  % BM: okay.
            bsxfun(@times, td, sinh(nrm_td) ./ nrm_td);
        
        % For those columns where the step is 0, replace y by x
        exclude = (nrm_td == 0);
        y(:, exclude) = x(:, exclude);

        y = trnsp(y);
    end
    

    M.retr = @retraction;
    % Retraction on the oblique manifold: we call the exponential directly.
    function y = retraction(x, d, t)
        if nargin < 3
            % t = 1;
            y = exponential(x, d);
        else
            y = exponential(x, d, t);
        end
    end
    
    M.log = @logarithm;
    function v = logarithm(x1, x2)
        x1 = trnsp(x1);
        x2 = trnsp(x2);
        
        theta = acosh(-sum(x1.*(bsxfun(@times, x2, g)),1));
        fstartheta = theta./sinh(theta);

        v = x2 - bsxfun(@times, x1, cosh(theta));
        % For very close points, dists is almost equal to norms, but
        % because they are both almost zero, the division above can return
        % NaN's. To avoid that, we force those ratios to 1.
        fstartheta(theta <= 1e-10) = 1;
        v = bsxfun(@times, v, fstartheta);
        
        % % Debug: another implementation
        % proj1 = x2 + bsxfun(@times, x1, sum(x1.*(bsxfun(@times, x2, g)),1));
        % norm1 = sqrt(sum(proj1.*(bsxfun(@times, proj1, g)),1));
        % v1 = bsxfun(@times, proj1, theta./norm1);
        % norm(v - v1)

        v = trnsp(v);

        % % Debug: the following should output zeros.
        % v2 = projection(x1, v);
        % norm(v(:) - v2(:))
        % M.dist(exponential(x1, v), x2)

    end

    M.hash = @(x) ['z' hashmd5(x(:))];
    
    M.rand = @myrand;
    function x = myrand() % BM: okay.
    	x1 = randn(n, m);
    	x0 = sqrt(1 + sum(x1.^2, 1));
    	x = [x0; x1]; 
    	x = trnsp(x);
    end
    
    M.randvec = @myrandvec;
    function u = myrandvec(x)
    	u = randn(size(x));
    	u = projection(x, u);
    	u = u / norm(u(:)); % Unit norm vector.
    end
    
    M.lincomb = @matrixlincomb;
    
    M.zerovec = @(x) trnsp(zeros(n+1, m));
    
    M.transp = @(x1, x2, d) M.proj(x2, d);
    
   
    % vec returns a vector representation of an input tangent vector which
    % is represented as a matrix. mat returns the original matrix
    % representation of the input vector representation of a tangent
    % vector. vec and mat are thus inverse of each other. They are
    % furthermore isometries between a subspace of R^nm and the tangent
    % space at x.
    vect = @(X) X(:);
    M.vec = @(x, u_mat) vect(trnsp(u_mat));
    M.mat = @(x, u_vec) trnsp(reshape(u_vec, [n+1, m]));
    M.vecmatareisometries = @() true;

end
