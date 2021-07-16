n=10000;
A = randn(n);
A = .5*(A+A');

manifold = spherefactory(n);

mycostfunction = @(x) -x'*(A*x);

problem.cost = mycostfunction; 
problem.M = manifold;

autogradfunc = autograd(problem);
problem.egrad = @(x) egradcompute(autogradfunc,x);


figure;
checkgradient(problem);

options.maxiter = 100;
[x, xcost, info] = steepestdescent(problem,[],options);         


% Display some statistics.
figure;
semilogy([info.iter], [info.gradnorm], '.-');
xlabel('Iteration #');
ylabel('Gradient norm');
title('Convergence of the trust-regions algorithm on the sphere');


