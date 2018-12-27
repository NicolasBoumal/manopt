% Dec. 16, 2018, NB
% Following up on "Cholesky QR-based retraction on the generalized Stiefel
% manifold" by Sato and Aihara, https://doi.org/10.1007/s10589-018-0046-7
%

clear all; close all; clc;

% The Cholesky baed approach might have numerical issues due to squaring,
% which leads to squaring the condition number. A modified Gram Schmidt
% executed directly in Euclidean space with G inner product resolve that
% issue (though admittedly it'd be a lot slower unless there is a C-mex
% version of it.)

n = 100;
p = 5;

F = randn(n);
B = F'*F;

M = stiefelgeneralizedfactory(n, p, B);
X = M.rand();
U = M.randvec(X);

Y = M.retr(X, U);
norm(Y'*B*Y - eye(p), 'fro')

% Modified Gram-Schmidt: input is a point X on the manifold and a tangent
% vector U at X. Output is Q = Retr_X(U).
Q_MGS  = stiefelgeneralized_retraction_MGS(B, X, U);
Q_MGS2 = stiefelgeneralized_retraction_MGS_twice(B, X, U);

% Mathematically, this Q corresponds to the "Cholesky QR" retraction,
% except it never squares the condition number, so it should in principle
% be numerically more robust (though it's probably slower because of the
% loop in Matlab.)
norm(Q_MGS'*B*Q_MGS - eye(p), 'fro')
norm(Q_MGS2'*B*Q_MGS2 - eye(p), 'fro')


%%
% Looking for a bad U
M = stiefelgeneralizedfactory(n, p, B);
N = tangentspherefactory(M, X); % with unit vectors, cond seems limited to sqrt(2)
% N = tangentspacefactory(M, X); % with any tangent vector
problem.M = N;
sqB = sqrtm(B);
problem.cost = @(U) -cond(sqB*(X+U)); % maximize the condition number of X+U in B-space
U = rlbfgs(problem, [], struct('maxtime', 500));