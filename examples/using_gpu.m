function using_gpu()
% Manopt example on how to use GPU with manifold factories that allow it.
%
% We are still working on this feature, and so far only few factories have
% been adapted to work on GPU. But the adaptations are rather easy. If
% there is a manifold you'd like to use on GPU, let us know via the forum
% on http://www.manopt.org, we'll be happy to help!
%
% See also: spherefactory complexcirclefactory

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Aug. 3, 2018.
% Contributors: 
% Change log: 

    % Construct a large problem to illustrate the use of GPU.
    % Below, we will compute a left-most eigenvector of A (symmetric).
    % On a particular test computer, we found that for n = 100, 1000, CPU
    % is faster, but for n = 10000, GPU tends to be 10x faster.
    n = 10000;
    A = randn(n);
    A = .5*(A+A');
    
    % First, setup and run the optimization problem on the CPU.
    problem.M = spherefactory(n, 1);   % sphere: nx1 vectors of norm 1
    problem.cost = @(x) .5*x'*(A*x);   % Rayleigh quotient to be minimized
    problem.egrad = @(x) A*x;          % could use caching to save here
    x0 = problem.M.rand();             % random initial guess
    tic_cpu = tic();
    x_cpu = trustregions(problem, x0); % run any solver
    time_cpu = toc(tic_cpu);
    
    % Then, move the data to the GPU, redefine the problem using the moved
    % data, activate the GPU flag in the factory, and run it again.
    A = gpuArray(A);
    problem.M = spherefactory(n, 1, true); % true is the GPU flag
    problem.cost = @(x) .5*x'*(A*x);       % code for cost and gradient are
    problem.egrad = @(x) A*x;              % basically the same, but operate
    x0 = gpuArray(x0);                     % on gpuArrays now.
    tic_gpu = tic();
    x_gpu = trustregions(problem, x0);
    time_gpu = toc(tic_gpu);
    
    fprintf('Total time CPU: %g\nTotal time GPU: %g\nSolution difference: %g\n', ...
            time_cpu, time_gpu, norm(x_cpu - x_gpu, 'fro'));
    
end
