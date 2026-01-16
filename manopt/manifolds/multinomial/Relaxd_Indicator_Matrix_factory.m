function M = Relaxd_Indicator_Matrix_factory(n, c, row,upper,lower)
% Manifold of n-by-c postive matrices such that row sum is "row" and lower < column sum < upper.
%  X > 0. X is positive
%  X1_n = row, row is a column positive vector of size n.
%  lower < X'1_c < upper , lower and upper are column positive vectors of size c.
%  Ensure that row > 0 , upper > 0 , lower > 0.
%
%
%  1.This code is capable of solving optimization problems on the Relaxed Indicator Matrix Manifold.
%  2.This code can efficiently solve optimization problems on the doubly stochastic manifold by setting lower = upper.
%
%
% Please cite the Manopt paper as well as the research papers:
%
% @article{Jinghui2025Riemannian,
% title={Riemannian Optimization on Relaxed Indicator Matrix Manifold},
%  author={Jinghui Yuan and Fangyuan Xie and Feiping Nie and Xuelong Li},
%  journal={arxiv},
%  year={2025}
%}
%
%
% The factory file extends the factory file
% multinomialdoublystochasticgeneralfactory 

    %Set the maximum number of iterations
    maxDSiters = min(1000, n*c); 

    %Check whether the dimensions of the input row-sum vector and column-bound vector are consistent.
    if size(row, 1) ~= n
        error('row should be a column vector of size n.');
    end
    if size(upper, 1) ~= c
        error('upper should be a column vector of size c.');
    end
    if size(lower, 1) ~= c
        error('lower should be a column vector of size c.');
    end
    
    %Return the name of the manifold.
    M.name = @() sprintf('%dx%d matrices with positive entries such that row sum is row and column sum is smaller than upper bigger then lower', n, c);

    %Return the dimension of the manifold, as proven in Theorem 1 of our paper.
    M.dim = @() (n-1)*c; 

    %Return the inner product, where the Euclidean inner product is the sum of the element-wise multiplications.
    M.inner = @iproduct; 
    function ip = iproduct(X,eta, zeta)
        ip = sum((eta(:).*zeta(:)));
    end

    %Return the length of the vector, which is the square root of its inner product with itself.
    M.norm = @(X,eta) sqrt(M.inner(X,eta, eta));

    M.typicaldist = @() n+c;

    %Return a point on the RIM manifold by randomly selecting a matrix and projecting it back onto the manifold using Dykstras algorithm, as Theorem 6 in our paper.
    M.rand = @random; 
    function X = random(X)
        Z = abs(randn(n, c));     
        X = Dykstras(Z, row, lower, upper, maxDSiters); 
    end

    %Return a tangent vector by randomly selecting a vector and projecting it onto the tangent space, as proven in Theorem 2 of our paper.
    M.randvec = @randomvec; 
    function eta = randomvec(X) 
        Z = randn(n, c);
        eta = ProjToTangent(Z);
    end

    %The function for the projection method, as proven in Theorem 2 of our paper.
    M.proj = @projection;  
    function etaproj = projection(X,eta) 
        etaproj = ProjToTangent(eta);
    end

    M.tangent = M.proj;

    %The tangent vector itself lies in the Euclidean space; return it as it is.
    M.tangent2ambient = @(X,eta) eta;

    %Return the Riemannian Gradient, which is the projection of the Euclidean Gradient, as proven in Theorem 2.
    M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(X,egrad) % projection of the euclidean gradient
        rgrad = ProjToTangent(egrad);
    end

    %The Retraction method, where Dykstras algorithm is used to map X + t*eta back onto the manifold.
    M.retr = @retraction;
    function Y = retraction(X, eta, t)
        if nargin < 3
            t = 1;
        end
        Y=Dykstras(X+t*eta, row, lower, upper, maxDSiters);
    end

    %Return the Riemannian Hessian, which is the projection of the Euclidean Hessian, as proven in Theorem 4.
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, eta)
        rhess = ProjToTangent(ehess);
    end

    %The function for the projection method, as proven in Theorem 2 of our paper.
    function P = ProjToTangent(X)
            c=size(X,2);
            P=X-1/c*X*ones(c,c);
    end


    M.hash = @(X) ['z' hashmd5(X(:))];
    M.lincomb = @matrixlincomb;
    M.zerovec = @(X) zeros(n, c);
    M.transp = @(X1, X2, d) ProjToTangent(d);
    M.vec = @(X, U) U(:);
    M.mat = @(X, u) reshape(u, n, c);
    M.vecmatareisometries = @() true;

%The algorithm for projecting onto the simplex can be found in reference [1]. The optimization problem is given by:
% min ||x - v||^2,s.t. x >= 0, sum(x) = k
% where k is typically set to 1 by default.
% Refer to Theorem 6 of our paper.
function [x,ft] = EProjSimplex_new(v, k)
if nargin < 2
    k = 1;
end

ft=1;
en = length(v);

v0 = v-mean(v) + k/en;
vmin = min(v0);
if vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        v1 = v0 - lambda_m;
        posidx = v1>0;
        npos = sum(posidx);
        g = -npos;
        f = sum(v1(posidx)) - k;
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > 100
            x = max(v1,0);
            break;
        end
    end
    x = max(v1,0);

else
    x = v0;
end
end

% Use Dykstra's algorithm for the projection, as detailed in reference [2], or refer to the proof section of Theorem 6 in our paper.
function [P] = Dykstras(M, a, b_l, b_u, N)
    if b_l==b_u 
        tol=1e-2;
    else
        tol=1e-1;
    end
    rng(1);
    [mn, mc] = size(M);
    P = M;
    z1 = zeros(mn, mc);
    z2 = zeros(mn, mc); 
    z3 = zeros(mn, mc); 
    
    for iter = 1:N
        for i = 1:mn
            prev_row = P(i, :) + z1(i, :);
            P(i, :) = EProjSimplex_new(prev_row, a(i));
            z1(i, :) = prev_row - P(i, :);
        end

        for j = 1:mc
            prev_col = P(:, j) + z2(:, j);
            current_sum = sum(prev_col);
            if current_sum >= b_l(j)
                z2(:, j) = 0;
                P(:, j) = prev_col;
            else
                delta = (b_l(j) - current_sum) / mn;
                new_col = prev_col + delta * ones(mn, 1);
                z2(:, j) = prev_col - new_col;
                P(:, j) = new_col;
            end
        end

        for j = 1:mc
            prev_col = P(:, j) + z3(:, j);
            current_sum = sum(prev_col);
            if current_sum <= b_u(j)
                z3(:, j) = 0; 
                P(:, j) = prev_col;
            else
                delta = (b_u(j) - current_sum) / mn;
                new_col = prev_col + delta * ones(mn, 1);
                z3(:, j) = prev_col - new_col;
                P(:, j) = new_col;
            end
        end

        if norm(P*ones(mc,1)-a, 'fro') < tol && all(P(:)>=-tol)
            disp(['Converged at iteration: ', num2str(iter)]);
            break;
        end
    end
end


end


%[1] The Constrained Laplacian Rank Algorithm for Graph-Based Clustering
%[2] Dykstra’s Algorithm, ADMM, and Coordinate Descent: Connections, Insights, and Extensions
