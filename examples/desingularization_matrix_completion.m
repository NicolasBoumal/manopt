function problem = desingularization_matrix_completion(run)

if nargin < 1
    run = false;
end

% Random data generation. First, choose the size of the problem.
% We will complete a matrix of size mxn of rank r.
m = 200;
n = 200;
r = 5;
alpha = rand();

% Generate a random mxn matrix A of rank r
L = randn(m, r);
R = randn(n, r);
A = L*R';

% Generate a random mask for observed entries: P(i, j) = 1 if the entry
% (i, j) of A is observed, and 0 otherwise.
fraction = 4*(m + n - r)*r/(m*n);
P = sparse(rand(m, n) <= fraction);

% Hence, we know the nonzero entries in PA:
PA = P.*A;

problem.M = desingularizationfactory(m, n, r, alpha);

% Define the problem cost function. The input X is a structure with
% fields U, S, V representing a rank r matrix as U*S*V'.
% f(X) = 1/2 * || P.*(X-A) ||^2
problem.cost = @cost;
function f = cost(X)
    % Note that it is very much inefficient to explicitly construct the
    % matrix X in this way. Seen as we only need to know the entries
    % of Xmat corresponding to the mask P, it would be far more
    % efficient to compute those only.
    Xmat = X.U*X.S*X.V';
    f = .5*norm(P.*Xmat - PA , 'fro')^2;
end

% Define the Euclidean gradient of the cost function, that is, the
% gradient of f(X) seen as a standard function of X.
% nabla f(X) = P.*(X-A)
problem.egrad = @egrad;
function G = egrad(X)
    % Same comment here about Xmat.
    Xmat = X.U*X.S*X.V';
    G = P.*Xmat - PA;
end

% This is optional, but it's nice if you have it.
% Define the Euclidean Hessian of the cost at X, along H, where H is
% represented as a tangent vector: a structure with fields K and Vp.
% This is the directional derivative of nabla f(X) at X along Xdot:
% nabla^2 f(X)[Xdot] = P.*Xdot
problem.ehess = @ehess;
function ehess = ehess(X, H)
    % Same comment here about explicitly constructing the ambient
    % vector as an mxn matrix Xdot: we only need its entries
    % corresponding to the mask P, and this could be computed
    % efficiently.
    Xdot = H.K*X.V' + X.U*X.S*H.Vp';
    ehess = P.*Xdot;
end

if run
    trustregions(problem);
end

end