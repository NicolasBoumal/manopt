    function[Xopt, Y] = testsimplexmatrices()
    clear all;
      rng(103)

    n = 2;
    k = 3;
    
    Y = zeros(n,n,k);
    for kk = 1:k
    	Y(:,:,kk) = randn(n,n) + 1i*randn(n,n);
    	Y(:,:,kk) = 0.5*(Y(:,:,kk) + (Y(:,:,kk))');
    	%Y(:,:,kk) = sparse(diag(diag(Y(:,:,kk)'*Y(:,:,kk))));
        %Y(:,:,kk) = sparse(((Y(:,:,kk)'*Y(:,:,kk))));
    end
    %Y(:,:,1) = -Y(:,:,1);
    % Ysum = sum(Y, 3);
    % Ysumsqrt = sqrtm(Ysum);
    % symm = @(X) .5*(X+X');
    % for kk = 1:k
    % 	Y(:,:,kk) = symm(Ysumsqrt\(Y(:,:,kk)/Ysumsqrt));
    % end



    problem.M = sympositivedefinitesimplexcomplexfactory(n, k);

    problem.cost = @cost;
    problem.egrad = @egrad;
    problem.ehess = @ehess;


    function f = cost(X)
    	f = 0;
    	for kk = 1:k
    		f = f + 0.5*norm((Y(:,:,kk) - X(:,:,kk)),'fro')^2;
    	end
    end

    function grad = egrad(X)
    	grad = zeros(size(X));
    	for kk = 1:k
    		grad(:,:,kk) = (X(:,:,kk) -  Y(:,:,kk));
    	end
    end

    function graddot = ehess(X, eta)
    	graddot = zeros(size(X));
    	graddot = eta;
    end


    checkgradient(problem);
    pause;
    checkhessian(problem);
    pause;
    
    options.linesearch = @linesearch_adaptive;
    options.maxinner = 30;
    options.maxiter = 100;
    Xopt = conjugategradient(problem,[],options);
    % Xopt = trustregions(problem,[],options);


    end
