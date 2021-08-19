clc; clear;
m = 42;
n = 60;
p = 5;
A = randn(m,n);

tuple.U = grassmannfactory(m, p);
tuple.V = grassmannfactory(n, p);
M = productmanifold(tuple);
    
problem.M = M;
problem.cost = @(X) cost(A,X);
 
problem = preprocessAD(problem);

figure;
checkgradient(problem);
figure;
checkhessian(problem);

    
 function f = cost(A,X)
    U = X.U;
    V = X.V;
    f = -.5* sum((U'*A*V).* (U'*A*V),'all');
 end