n = 200;
m = 2000;
p = 10;
q = 0.3;
X = rand(n,m);
[U, ~, ~] = svd(X);
U = U(:,1:p);
X = U * U' * X;
P = randperm(size(X, 2));
outliers = q * m;  
X(:, P(1:outliers)) = 30*randn(n, outliers);
epsilon = 1e-6;
    
manifold = grassmannfactory(n, p);  
problem.M = manifold;
problem.cost = @(U) mycostfunction(U,X,epsilon);


autogradfunc = autograd(problem);
problem.egrad = @(x) egradcompute(autogradfunc,x);
autohessfunc = autohess(problem);
%problem.ehess = @(x,xdot) ehesscompute(problem,x,xdot);
problem.ehess = @(x,xdot,store) ehesscompute_new(problem,x,xdot,store);


figure;
checkgradient(problem);
figure
checkhessian(problem);

U2 = rand(n,p);
[U2, Ucost, info] = trustregions(problem,U2);  
figure;
semilogy([info.iter], [info.gradnorm], '.-');
xlabel('Iteration #');
ylabel('Gradient norm');
title('Convergence of the trust region algorithm');

function value = mycostfunction(U,X,epsilon)

    vecs = U*(U'*X) - X;
    sqnrms = sum(vecs.^2, 1);
    vals = sqrt(sqnrms + epsilon^2) - epsilon;
    value = mean(vals);
        
end
