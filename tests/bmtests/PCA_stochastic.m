function PCA_stochastic()
% Example of use of the stochastic gradient algorithm in Manopt.

    % We get n samples in R^d
    d = 100;
    n = 100000;
    A = randn(n, d)*diag([[15 10 5], ones(1, d-3)]);

    % Setup the problem of computing PCA on the dataset A, that is,
    % computing the k top eigenvectors of A'*A:
    k = 3;
    
    problem.M = stiefelfactory(d, k);
    problem.ncostterms = n;
    problem.partialegrad = @partialegrad;
    function G = partialegrad(X, sample)
        Asample = A(sample, :);
        G = -Asample'*(Asample*X);
        G = G / n;
    end

%     problem.cost = @(X) -.5*norm(A*X, 'fro')^2 / n;
%     checkgradient(problem); pause;

    options.statsfun = statsfunhelper('metric', @(X) norm(A*X, 'fro'));
    options.maxiter = 200;
    options.batchsize = 10;
%     options.stepsize_type = 'decay';
    options.stepsize_init = 1e2;
    options.stepsize_lambda = 1e-3;
    options.checkperiod = 10;
    [X, info] = stochasticgradient(problem, [], options); %#ok<ASGLU>
    
    plot([info.iter], [info.metric], '.-');
    
    [V, ~] = svds(A', k);
    hold all;
    bound = norm(A*V, 'fro');
    plot([info.iter], bound*ones(size([info.iter])), '--');
    
%     keyboard;
    
end
