n=1000;
A = randn(n);
A = .5*(A+A');
    
manifold = spherefactory(n);

mycostfunction = @(x) -x'*(A*x);

problem.cost = mycostfunction; 
problem.M = manifold;

autogradfunc = autograd(problem);
problem.egrad = @(x) egradcompute(autogradfunc,x);
autohessfunc = autohess(problem);
problem.ehess = @(x,xdot) ehesscompute(autohessfunc,x,xdot);


figure;
checkgradient(problem);
figure;
checkhessian(problem);

%options.maxiter = 100;
[x, xcost, info] = trustregions(problem);         


% Display some statistics.
figure;
semilogy([info.iter], [info.gradnorm], '.-');
xlabel('Iteration #');
ylabel('Gradient norm');
title('Convergence of the trust-regions algorithm on the sphere');


