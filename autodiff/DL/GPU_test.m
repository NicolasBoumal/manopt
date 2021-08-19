clear;clc;
p = 3;
n = 5000;
A = randn(n);
A = A+A';
inner = @(U, V) U(:)'*V(:);

% CPU
problem_cpu.M = grassmannfactory(n, p, 1); 
problem_cpu.cost = @(X) .5*inner(X, A*X); 
problem_cpu = preprocessAD(problem_cpu);
X0 = problem_cpu.M.rand();

tic_cpu = tic();
X_cpu = trustregions(problem_cpu,X0);
time_cpu = toc(tic_cpu);

% GPU
A = gpuArray(A);
X0 = gpuArray(X0);
problem_gpu.M = grassmannfactory(n, p, 1, true); 
problem_gpu.cost = @(X) .5*inner(X, A*X); 
problem_gpu = preprocessAD(problem_gpu);

tic_gpu = tic();
X_gpu = trustregions(problem_gpu,X0);
time_gpu = toc(tic_gpu);

fprintf('Total time CPU: %g\nTotal time GPU: %g\nSolution difference: %g\n', ...
            time_cpu, time_gpu, norm(X_cpu - X_gpu, 'fro'));
