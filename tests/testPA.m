% svd_warmstart_02PA - PA Absil - Started Tue 11 Jun 2013
%    Modifs.
% svd_warmstart_01PA - PA Absil - Started Tue 11 Jun 2013
%    Matlab code for testing warm-start truncated SVD, using the Manopt toolbox.
%    This script has been tested with Manopt 1.0.2.

% The rest of this script is modeled on the example found at
% http://www.manopt.org/tutorial.html

% The underlying approach to the extreme eigenvalue problem is described in
% http://dx.doi.org/10.1016/j.cam.2005.10.006
% The concept is rather basic: use a black-box Riemannian optimization method to
% maximize a generalized Rayleigh quotient (see problem.cost below) on a
% Grassmann manifold.  

% For info, a more efficient manifold-inspired method for extreme
% symmetric eigenvalue problems (and thus for truncated SVDs) is IRTR,
% found in Anasazi within Sandia's Trilinos:
% http://trilinos.sandia.gov/packages/docs/r10.2/packages/anasazi/doc/html/classAnasazi_1_1IRTR.html 

% Install Manopt:
% Go to http://www.manopt.org/, click on Download, unpack the zip file,
% have a look at README.txt.

% In the Matlab session, import the Manopt toolbox:
% importmanopt;  % importmanopt.m is located in Manopt_1.0.2/manopt/

% Check installation by running an example:
% basicexample

% Generate the problem data:
m = 100;  % number of pixels in the speckle removal application (in
          % practice, around 500*500)
n = 10;   % number of images in the speckel removal application (in
          % practice, around 500)
D = randn(m,n);  % data matrix

% Number of sought components:
p = 5;

% Magnitude of the perturbation that will be applied to the data to test
% warm start:
mag_pert = 1e-3;

% Set a stopping criterion based on the norm of the gradient:
options.tolgradnorm = 1e-3;
% See stoppingcriterion.m for more information.

% Create the problem structure:
manifold = grassmannfactory(n,p);
problem.M = manifold;



% ******* Cold-start truncated SVD *******

% We want to compute bases U and V of the p-dimensional left and right dominant
% singular spaces of D.
% Equivalently, we compute a basis V of the p-dimensional dominant
% eigenspace of A = D'*D, and then get U as D*V.
A = D'*D;
% If m was smaller than n, then it would be advisable to compute a basis
% U of the p-dimensional dominant eigenspace of A = D*D', and then get V
% as D'*U.

% Another A for testing purposes, where the solution is well known:
%A = diag([n:-1:1]);

% Define the problem cost function and its gradient:
problem.cost = @(x) -trace(x'*A*x); 
% The minus sign appears because we want to maximize the generalized
% Rayleigh quotient x\mapsto tr(x^TAx), whereas the Manopt solvers
% minimize the given cost function.
problem.grad = @(x) manifold.egrad2rgrad(x, -2*A*x);
% -2Ax is the Euclidean gradient of the cost function.

% If D is very sparse and n is not so small, then it may be beneficial to
% avoid constructing A = D'*D:
%problem.cost = @(x) -trace(x'*D'*D*x); 
%problem.grad = @(x) manifold.egrad2rgrad(x, -2*D'*D*x);

% Numerically check gradient consistency:
%checkgradient(problem);

% Solve the Riemannian optimization problem:
tic
%[x xcost info] = trustregions(problem,[],options);
[x xcost info] = conjugategradient(problem,[],options);
% In the present context, I'd recommend using conjugategradient rather than
% trustregions as a solver, because the latter tends to "overconverge",
% i.e., reach a (much) higher precision than requested. This is due to the
% inner iteration, in which the (outer) stopping criterion is not checked.
time_cold = toc;

% Display some statistics:
figure;
semilogy([info.iter], [info.gradnorm], '.-');



% ******* Warm-start truncated SVD *******

% We could test warm start by keeping D (and A) unchanged and feeding a
% perturbed solution as initial condition:
%x_init = orth(x+mag_pert*randn(size(x)));  % perturbed solution
%problem_pert = problem;
% orth returns an orthonormal basis of the range of its argument. This is
% necessary because Manopt represents subspaces by orthonormal bases.

% Other test: we slightly perturb matrix D and feed the solution obtained
% for the original D:
D_pert = D + mag_pert * randn(size(D));
x_init = x;
A_pert = D_pert' * D_pert;
problem_pert = problem;
problem_pert.cost = @(x) -trace(x'*A_pert*x); 
problem_pert.grad = @(x) manifold.egrad2rgrad(x, -2*A_pert*x);

% Solve with warm start:
tic
%[x_ws xcost_ws info_ws] = trustregions(problem,x_pert,options);
[x_ws xcost_ws info_ws] = conjugategradient(problem_pert,x_init,options);
time_warm = toc;

% Display some statistics:
figure;
semilogy([info_ws.iter], [info_ws.gradnorm], '.-');


% ******* Display timing comparison between cold and warm *******
fprintf('\n');
fprintf('With cold start, the solver returned after %f [s]\n', time_cold);
fprintf('With warm start, the solver returned after %f [s]\n', time_warm);
