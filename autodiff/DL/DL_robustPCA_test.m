X = rand(2, 1)*(1:30) + .05*randn(2, 30).*[(1:30);(1:30)];
P = randperm(size(X, 2));
outliers = 10;  
X(:, P(1:outliers)) = 30*randn(2, outliers);
epsilon = 1e-5;
    
    
manifold = grassmannfactory(2, 1);
problem.M = manifold;
problem.cost = @(U) mycostfunction(U,X,epsilon);
autogradfunc = autograd(problem);
problem.egrad = @(x) egradcompute(autogradfunc,x);
    
figure;
checkgradient(problem);

[U, ~, ~] = svds(X, 1);
[U, Ucost, info] = steepestdescent(problem,U);  
figure;
semilogy([info.iter], [info.gradnorm], '.-');
xlabel('Iteration #');
ylabel('Gradient norm');
title('Convergence of the steepest descent algorithm');

function value = mycostfunction(U,X,epsilon)

    vecs = U*(U'*X) - X;
    sqnrms = sum(vecs.^2, 1);
    vals = sqrt(sqnrms + epsilon^2) - epsilon;
    value = mean(vals);
        
end
