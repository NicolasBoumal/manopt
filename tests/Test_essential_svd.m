%Sample solution of an optimization problem on the essential manifold
%Solves the problem \sum_{i=1}^N ||E_i-A_i||^2, where E_i are essential
%matrices
function Test_essential_svd
%Make data for the test
N=2;    %number of matrices to process in parallel
A=multiprod(multiprod(randrot(3,N),hat3([0;0;1])),randrot(3,N));

%Make the manifold
M=essentialfactory(N);
problem.M=M;

%Function of the essential matrix E and Euclidean gradient and Hessian
ef=@(E) 0.5*sum(multitrace(multiprod(multitransp(E-A),(E-A))));
egradf=@(E) E-A;
ehessf=@(E,S) S;

%Function on the essential manifold and the Riemannian gradient and Hessian
rf=@(X) M.ef2rf(X,ef);
rgradf=@(X) M.egradE2rgrad(X,egradf);
rhessf=@(X,S) M.ehessE2rhess(X,egradf,ehessf,S);

%Setup MANOPT problem
problem.cost=rf;
problem.grad=rgradf;
problem.hess=rhessf;

% Numerically check the differentials.
checkgradient(problem); pause;
checkhessian(problem); pause;

%Solve the problem
X=trustregions(problem);

disp('Distance between original matrices and decompositions')
disp(sqrt(ef(M.E(X))*2))

