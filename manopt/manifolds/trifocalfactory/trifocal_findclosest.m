function trifocal_findclosest
% Sample solution of an optimization problem on the trifocal manifold.
%
% Solves the problem \sum_{i=1}^k ||T_i-A_i||^2, where T_i,A_i are trifocal tensors.
% Trifocal tensors are used in computer vision to descrive relations
% between projection of points and/or lines in three views.
%
% Note: the trifocalfactory file uses a quotient  representation to
% work with trifocal tensors. On the other hand, from a user point of 
% view, it is convenient to use the T representation  (a tensor of size
% 3-by-3-by-3) to give cost, gradient, and Hessian  information. To this end, we
% provide auxiliary files trifocal_costT2cost, trifocal_egradT2egrad, and
% trifocal_ehessE2rhess.
%
% See also: tricocalfactory trifocal_costT2cost trifocal_egradT2egrad
% trifocal_ehessT2rhess

% define manifold
k = 3;
M  = trifocalfactory(k);
problem.M = M;

% generate some random data
X =  M.rand();
A = M.T(X);

% Function handles of the trifocal tensor T and Euclidean gradient and Hessian
costT  = @(T) .5*norm(T(:)-A(:),'fro')^2;
egradT = @(T) T - A;
ehessT = @(T, U) U;

% Manopt descriptions
problem.cost = @cost;
function val = cost(X)
    val = trifocal_costT2cost(X, costT); % Cost
end

problem.egrad = @egrad;
function g = egrad(X)
    g = trifocal_egradT2egrad(X, egradT); % Converts gradient in T to X.
end


problem.hess = @hess;
function gdot = hess(X, dX)
    gdot = trifocal_ehessT2rhess(X,egradT,ehessT,dX); % Converts hessian in T to X.
end

% Numerically check gradient and hessian consistency 
checkgradient(problem);pause
checkhessian(problem);pause

 % Initialization (close enough to the true value) 
 X0 = M.exp(X,randn*M.randvec(X)); 

% Solve
[Xm, ~, info] = trustregions(problem,X0);


% Display some statistics.
figure,
semilogy([info.iter], [info.cost], '.-');
xlabel('Iteration number');
ylabel('Cost');

fprintf('Distance between original tensors and decompositions is %e \n', costT(problem.M.T(Xm)));




end