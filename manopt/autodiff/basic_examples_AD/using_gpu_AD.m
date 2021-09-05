function using_gpu_AD()
% Manopt example on how to use GPU to compute the egrad and the ehess via AD.
%
% This file is basically the same as using_gpu.m.
%
% See also: using_gpu

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Xiaowen Jiang, Aug. 21, 2021.
% Contributors: 
% Change log: 

    % Verify that the deep learning tool box was installed
    assert(exist('dlarray', 'file') == 2, ['Deep learning tool box is '... 
    'needed for automatic differentiation.\n Please install the'...
    'latest version of the deep learning tool box and \nupgrade to Matlab'...
    ' R2021b if possible.'])

    if exist('OCTAVE_VERSION', 'builtin')
        warning('manopt:usinggpu', 'Octave does not handle GPUs at this time.');
        return;
    end

    if gpuDeviceCount() <= 0
        warning('manopt:usinggpu', 'No GPU available: cannot run example.');
        return;
    end

    % Construct a large problem to illustrate the use of GPU.
    % Below, we will compute p left-most eigenvectors of A (symmetric).
    % On a particular test computer.
    p = 3;
    n = 10000;
    A = randn(n);
    A = A+A';
    
    inner = @(U, V) U(:)'*V(:);
    
    % First, setup and run the optimization problem on the CPU.
    problem_cpu.M = grassmannfactory(n, p, 1); % 1 copy of Grassmann(n, p)
    problem_cpu.cost = @(X) .5*inner(X, A*X);  % Rayleigh quotient to be minimized
    problem_cpu = manoptAD(problem_cpu);       % Obtain the egrad and ehess via AD 
    X0 = problem_cpu.M.rand();                 % Random initial guess
    tic_cpu = tic();
    X_cpu = trustregions(problem_cpu, X0);     % run any solver
    time_cpu = toc(tic_cpu);
    
    % Then, move the data to the GPU, redefine the problem 
    % activate the GPU flag in the factory, and run it again.
    A = gpuArray(A);
    problem_gpu.M = grassmannfactory(n, p, 1, true); % true is the GPU flag;
    problem_gpu.cost = @(X) .5*inner(X, A*X);        % Code for cost
    problem_gpu = manoptAD(problem_gpu);         % Work on gpu now            
    X0 = gpuArray(X0);
    tic_gpu = tic();
    X_gpu = trustregions(problem_gpu, X0);
    time_gpu = toc(tic_gpu);
    
    fprintf('Total time CPU: %g\nTotal time GPU: %g\nSolution difference: %g\n', ...
            time_cpu, time_gpu, norm(X_cpu - X_gpu, 'fro'));
    
end
