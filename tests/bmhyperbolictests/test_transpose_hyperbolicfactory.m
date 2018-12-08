function xopt = test_transpose_hyperbolicfactory()

n = 10;
N = 5;

transposed = 1;

if transposed % BM: okay
        trnsp = @(X) X';
else
        trnsp = @(X) X;
end

problem.M = hyperbolicfactory(n, N, transposed);

Y = randn(n+1, N);

problem.cost = @cost;
problem.egrad = @egrad;
problem.ehess = @ehess;

	function f = cost(x)
		x = trnsp(x);
		f =  0.5*norm(x - Y,'fro')^2;
	end

	function grad = egrad(x)
		x = trnsp(x);
		grad = zeros(size(x));
		grad = x- Y;
		grad = trnsp(grad);
	end

	function graddot = ehess(x, u)
		x = trnsp(x);
		u = trnsp(u);

		graddot = u;

		graddot = trnsp(graddot);
	end

% Check gradient
checkgradient(problem);
pause;
checkhessian(problem);

xopt = manoptsolve(problem);

end