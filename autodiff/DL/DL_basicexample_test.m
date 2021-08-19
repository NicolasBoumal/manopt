clc;clear;

rng(2019)
n=1000;
A = randn(n);
A = .5*(A+A');    
A = gpuArray(A);

%manifold = stiefelcomplexfactory(n,1);
rng(2019)
manifold = stiefelfactory(n,1,1,true);

mycostfunction = @(x) -x'*(A*x);
%mycostfunction = @(x) -creal(cprod(ctransp(x),cprod(A,x)));

problem.cost = mycostfunction; 
problem.M = manifold;   
problem = preprocessAD(problem);
% problem.egrad = @(x) -2*(A*x);
% problem.ehess = @(x,xdot) -2*(A*xdot);



figure;
checkgradient(problem);
figure;
checkhessian(problem);

%options.maxiter = 200;
rng(2019)
[x, xcost, info] = trustregions(problem);         

