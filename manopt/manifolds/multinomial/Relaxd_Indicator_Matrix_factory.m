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

    maxDSiters = min(1000, n*c); % Ideally it should be supplid by user. 
    if size(row, 1) ~= n
        error('row should be a column vector of size n.');
    end

    if size(upper, 1) ~= c
        error('upper should be a column vector of size c.');
    end

    if size(lower, 1) ~= c
        error('lower should be a column vector of size c.');
    end

    M.name = @() sprintf('%dx%d matrices with positive entries such that row sum is row and column sum is smaller than upper bigger then lower', n, c);

    M.dim = @() (n-1)*c; 

    M.inner = @iproduct; 
    function ip = iproduct(X,eta, zeta)
        ip = sum((eta(:).*zeta(:)));
    end

    M.norm = @(X,eta) sqrt(M.inner(X,eta, eta));

    M.typicaldist = @() n+c;

    M.rand = @random; 
    function X = random(X)
        Z = abs(randn(n, c));     
        X = Dykstras(Z, row, lower, upper, maxDSiters); 
    end

    M.randvec = @randomvec; 
    function eta = randomvec(X) 
        Z = randn(n, c);
        eta = ProjToTangent(Z);
    end

    M.proj = @projection;  
    function etaproj = projection(X,eta) 
        etaproj = ProjToTangent(eta);
    end

    M.tangent = M.proj;
    M.tangent2ambient = @(X,eta) eta;

    M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(X,egrad) % projection of the euclidean gradient
        rgrad = ProjToTangent(egrad);
    end

    M.retr = @retraction;
    function Y = retraction(X, eta, t)
        if nargin < 3
            t = 1;
        end
        Y=Dykstras(X+t*eta, row, lower, upper, maxDSiters);
    end


    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, eta)
        rhess = ProjToTangent(ehess);
    end

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


function [P] = Dykstras(M, a, b_l, b_u, N)
% Dykstra's algorithm for projecting onto intersection of convex sets
% Input:
%   M: initial matrix
%   a: vector for simplex constraints (feasible set 1)
%   b_l: lower bounds for column sum constraints (feasible set 2)
%   b_u: upper bounds for column sum constraints (feasible set 3)
%   N: maximum number of iterations
% Output:
%   P: projected matrix
%   obj: objective value (optional, for tracking convergence)
if b_l==b_u 
    tol=1e-2;
else
    tol=1e-1;
end
rng(1);
[mn, mc] = size(M);
P = M;
z1 = zeros(mn, mc); % Residuals for feasible set 1
z2 = zeros(mn, mc); % Residuals for feasible set 2
z3 = zeros(mn, mc); % Residuals for feasible set 3

for iter = 1:N

        % Project onto feasible set 1 (simplex constraints)
    for i = 1:mn
        P(i, :) = EProjSimplex_new(P(i, :) + z1(i, :), a(i));
        z1(i, :) = P(i, :) + z1(i, :) - P(i, :); % Update residual
    end
    % Project onto feasible set 2 (lower bound constraints)
    for j = 1:mc
        if sum(P(:, j) + z2(:, j)) >= b_l(j)
            z2(:, j) = P(:, j) + z2(:, j) - P(:, j); % No adjustment needed
        else
            delta = (b_l(j) - sum(P(:, j) + z2(:, j))) / mn;
            P(:, j) = P(:, j) + z2(:, j) + delta * ones(mn, 1);
            z2(:, j) = P(:, j) + z2(:, j) - P(:, j); % Update residual
        end
    end

    % Project onto feasible set 3 (upper bound constraints)
    for j = 1:mc
        if sum(P(:, j) + z3(:, j)) <= b_u(j)
            z3(:, j) = P(:, j) + z3(:, j) - P(:, j); % No adjustment needed
        else
            delta = (b_u(j) - sum(P(:, j) + z3(:, j))) / mn;
            P(:, j) = P(:, j) + z3(:, j) + delta * ones(mn, 1);
            z3(:, j) = P(:, j) + z3(:, j) - P(:, j); % Update residual
        end
    end



    % Convergence check (optional, based on residual norms)
    if norm(P*ones(mc,1)-a, 'fro') < tol && all(P(:)>=-tol)
        disp(['Converged at iteration: ', num2str(iter)]);
        break;
    end
end
end


end



